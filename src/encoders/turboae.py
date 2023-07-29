from typing import Tuple, Dict, Any, TypedDict
import abc

import torch
import torch.nn.functional as F

from ..utils import (
    DeviceManager,
    DEFAULT_DEVICE_MANAGER,
    enumerate_binary_inputs_chunked,
    enumerate_binary_inputs,
    base_2_accumulator,
    data_gen,
    NamedTensor,
    dynamic_get,
    dynamic_slice,
    MaskedTensor,
)
from ..graphs import (
    nonrecursive_convolutional_factors,
    nonrecursive_turbo_graph,
    InferenceGraph,
)
from ..constants import INPUT_SYMBOL
from ..interleavers import FixedPermuteInterleaver
from ..neural_utils import SameShapeConv1d
from ..channels import NoisyChannel
from ..modulation import Modulator

from .encoder import Encoder
from .block_encoder import CodebookEncoder
from .convolutional_encoder import GeneralizedConvolutionalEncoder
from .turbo import StreamedTurboEncoder
from ..engine import ResultsProcessor, TqdmProgressBar


class ENCBase(Encoder):
    def __init__(self, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER):
        super(ENCBase, self).__init__(device_manager=device_manager)

        self.this_device = self.device_manager.device
        self.enc_act_name = "elu"

    def enc_act(self, inputs):
        if self.enc_act_name == "tanh":
            return F.tanh(inputs)
        elif self.enc_act_name == "elu":
            return F.elu(inputs)
        elif self.enc_act_name == "relu":
            return F.relu(inputs)
        elif self.enc_act_name == "selu":
            return F.selu(inputs)
        elif self.enc_act_name == "sigmoid":
            return F.sigmoid(inputs)
        elif self.enc_act_name == "linear":
            return inputs
        else:
            return inputs

    @property
    @abc.abstractmethod
    def input_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_output_channels(self):
        pass

    def is_codeword(self, y_hat: torch.Tensor) -> Tuple[torch.BoolTensor, torch.Tensor]:
        raise NotImplementedError

    def to_codebook(self, dtype=torch.float32, chunk_size=2**18) -> CodebookEncoder:
        all_input_iter = enumerate_binary_inputs_chunked(
            window=self.input_size,
            dtype=torch.int8,
            chunk_size=chunk_size,
            device=self.device_manager.device,
        )
        codebook = torch.empty(
            (2**self.input_size, self.num_output_channels * self.input_size),
            dtype=dtype,
            device=self.device_manager.device,
        )
        start = 0
        for inputs in all_input_iter:
            end = start + inputs.shape[0]
            codebook[start:end] = torch.reshape(
                self(inputs, dtype=dtype), (inputs.shape[0], -1)
            )
            start = end


        return CodebookEncoder(codebook=codebook, device_manager=self.device_manager)

    def settings(self) -> Dict[str, Any]:
        return {"enc_act": self.enc_act_name, "type": self.__class__.__name__}


class ENC_interCNN(ENCBase):
    def __init__(
        self,
        enc_num_layer: int,
        enc_num_unit: int,
        enc_kernel_size: int,
        interleaver: FixedPermuteInterleaver,
        first_pad=False,
        front_pad=False,
        binarize=False,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ):
        # turbofy only for code rate 1/3
        super(ENC_interCNN, self).__init__(device_manager=device_manager)
        code_rate_k = 1
        # XXX: This changes behavior! Padding now puts -1 instead of 0
        pad_value = -1.0
        # Encoder - automatically using elu activation
        self.enc_cnn_1 = SameShapeConv1d(
            num_layer=enc_num_layer,
            in_channels=code_rate_k,
            out_channels=enc_num_unit,
            kernel_size=enc_kernel_size,
            first_pad=first_pad,
            front_pad=front_pad,
            pad_value=pad_value,
        )

        self.enc_cnn_2 = SameShapeConv1d(
            num_layer=enc_num_layer,
            in_channels=code_rate_k,
            out_channels=enc_num_unit,
            kernel_size=enc_kernel_size,
            first_pad=first_pad,
            front_pad=front_pad,
            pad_value=pad_value,
        )

        self.enc_cnn_3 = SameShapeConv1d(
            num_layer=enc_num_layer,
            in_channels=code_rate_k,
            out_channels=enc_num_unit,
            kernel_size=enc_kernel_size,
            first_pad=first_pad,
            front_pad=front_pad,
            pad_value=pad_value,
        )

        self.enc_linear_1 = torch.nn.Linear(enc_num_unit, 1)
        self.enc_linear_2 = torch.nn.Linear(enc_num_unit, 1)
        self.enc_linear_3 = torch.nn.Linear(enc_num_unit, 1)
        self.front_pad = front_pad
        self.first_pad = first_pad

        self.interleaver = interleaver

        self.binarize = binarize

        self.register_buffer("enc_num_layer", torch.tensor(enc_num_layer))
        self.register_buffer("code_rate_k", torch.tensor(code_rate_k))
        self.register_buffer("enc_num_unit", torch.tensor(enc_num_unit))
        self.register_buffer("enc_kernel_size", torch.tensor(enc_kernel_size))

        self.to(self.device_manager.device)

        self.mean = 0
        self.std = 1
        self.mean_std_initialized = False

        self._base_2_accumulator = base_2_accumulator(
            self.window, device=self.device_manager.device
        ).float()

    def set_parallel(self):
        self.enc_cnn_1 = torch.nn.DataParallel(self.enc_cnn_1)
        self.enc_cnn_2 = torch.nn.DataParallel(self.enc_cnn_2)
        self.enc_cnn_3 = torch.nn.DataParallel(self.enc_cnn_3)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)
        self.enc_linear_3 = torch.nn.DataParallel(self.enc_linear_3)

    @property
    def input_size(self):
        return self.interleaver.input_size

    @property
    def window(self):
        assert self.enc_num_layer > 0
        return self.enc_kernel_size + (self.enc_kernel_size - 1) * (
            self.enc_num_layer - 1
        )

    @property
    def delay(self):
        # if not self.first_pad:
        #     # raise print("Undefined delay if we have boundary codes")
        return self.enc_cnn_1.make_pad_tuple(self.enc_cnn_1._out_shape_reduced_count())[
            1
        ]

    @torch.no_grad()
    def compute_mean_std(self, num_samples: int = 50000, batch_size: int = 1000):
        num_steps = (num_samples + batch_size - 1) // batch_size
        print(
            f"Computing mean and std of encoder over {num_steps * batch_size} samples with {num_steps} batches of batch size {batch_size}."
        )
        pbar = TqdmProgressBar()
        pbar.new_experiment(total=num_steps)
        results_processor = ResultsProcessor([pbar])
        for data in data_gen(
            self.input_size,
            num_steps=num_steps,
            batch_size=batch_size,
            device_manager=self.device_manager,
        ):
            for tensor in data:
                out = torch.flatten(self(tensor))
                results_processor.update({"output": out})
        results = results_processor.results
        print(
            f"Computed mean {results['output__mean']} with error +/-{results['output__err']}"
        )
        print(f"Computed std {results['output__std']}")
        return {"mean": results["output__mean"], "std": results["output__std"]}

    @torch.no_grad()
    def compute_mean_std_(self, num_samples: int = 50000, batch_size: int = 1000):
        results = self.compute_mean_std(num_samples=num_samples, batch_size=batch_size)
        self.mean = results["mean"]
        self.std = results["std"]
        self.mean_std_initialized = True

    @staticmethod
    def preprocess_input_state_dict(state_dict):
        discard_list = ["deinterleaver.reverse_p_array", "interleaver.p_array"]
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
            if k not in discard_list
        }
        return state_dict

    def pre_initialize(self, state_dict):
        state_dict = self.preprocess_input_state_dict(state_dict)
        # Vestige from pre_initialize method of the turboae_decoder
        # check_ks = []
        # for k in [k for k in check_ks if k in state_dict]:
        #     expected = state_dict.pop(k).item()
        #     actual = getattr(self, k[1:])
        #     assert (
        #         actual == expected
        #     ), f"Check {k} failed: self.{k[1:]}={actual} but state_dict[{k}]={expected}"
        check_ks_state_dict = ["interleaver.permutation", "interleaver.depermutation"]
        current_state_dict = self.state_dict()
        for k in [k for k in check_ks_state_dict if k in state_dict]:
            expected = state_dict[k]
            actual = current_state_dict[k]
            assert torch.all(
                expected == actual
            ), f"Check {k} failed: incoming={expected} but current={actual}."
        try:
            self.load_state_dict(state_dict=state_dict)
        except Exception as e:
            print(e)
            print("Trying with strict=False")
            self.load_state_dict(state_dict=state_dict, strict=False)

    @property
    def num_output_channels(self):
        return 3

    def forward(
        self, inputs, dtype=torch.float, interleave=True, table=True, constrain=True
    ):
        if table and self.first_pad:
            # Use the table instead for more stable (???) training
            table_noni, table_i = self.to_table(constrain=constrain)
            pad_tuple = self.enc_cnn_1.make_pad_tuple(
                self.enc_cnn_1._out_shape_reduced_count()
            ) + (0, 0)
            x_noni = GeneralizedConvolutionalEncoder._conv_encode(
                msg=F.pad(inputs, pad=pad_tuple),
                table=table_noni,
                _base_2_accumulator=self._base_2_accumulator,
                device=self.device_manager.device,
            )

            if interleave:
                inputs_int = self.interleaver(inputs)
            else:
                inputs_int = inputs

            x_i = GeneralizedConvolutionalEncoder._conv_encode(
                msg=F.pad(inputs_int, pad=pad_tuple),
                table=table_i,
                _base_2_accumulator=self._base_2_accumulator,
                device=self.device_manager.device,
            )

            x_tx = torch.cat([x_noni, x_i], dim=2)

            return x_tx.to(dtype)
        else:
            inputs = 2.0 * inputs.float()[..., None] - 1.0
            x_sys = self.enc_cnn_1(inputs)
            x_sys = self.enc_act(self.enc_linear_1(x_sys))

            x_p1 = self.enc_cnn_2(inputs)
            x_p1 = self.enc_act(self.enc_linear_2(x_p1))

            if interleave:
                x_sys_int = self.interleaver(inputs)
            else:
                x_sys_int = inputs
            x_p2 = self.enc_cnn_3(x_sys_int)
            x_p2 = self.enc_act(self.enc_linear_3(x_p2))

            x_tx = torch.cat([x_sys, x_p1, x_p2], dim=2)

            if self.binarize:
                return torch.sign(x_tx)
            else:
                return (x_tx.to(dtype) - self.mean) / self.std

    def to_conv_code(
        self,
        constrain=True,
        no_delay=False,
    ):
        table_noni, table_i = self.to_table(constrain=constrain)
        constraint = "opt_unit_power" if constrain else None

        delay = 0 if no_delay else self.delay
        encoder1 = GeneralizedConvolutionalEncoder(
            table=table_noni,
            num_steps=self.interleaver.input_size,
            constraint=constraint,
            delay=delay,
            device_manager=self.device_manager,
        )
        encoder2 = GeneralizedConvolutionalEncoder(
            table_i,
            num_steps=self.interleaver.input_size,
            constraint=constraint,
            delay=delay,
            device_manager=self.device_manager,
        )
        encoder1.apply_constraint_()
        encoder2.apply_constraint_()
        encoder1.update()
        encoder2.update()
        return StreamedTurboEncoder(
            noninterleaved_encoder=encoder1,
            interleaved_encoder=encoder2,
            interleaver=self.interleaver,
            device_manager=self.device_manager,
        )

    FullTable = TypedDict(
        "FullTable",
        {"front": MaskedTensor, "main": torch.FloatTensor, "end": MaskedTensor},
    )

    def to_full_table(
        self,
    ) -> FullTable:
        bin_in = enumerate_binary_inputs(
            window=self.window, dtype=torch.float, device=self.device_manager.device
        )
        # If we did front pad, then the value we want will be in the last output.
        # If we did even padding on both front and back (so with delay), then
        # the value we want will be in the middle output
        output_ind = self.enc_cnn_1.make_pad_tuple(
            self.enc_cnn_1._out_shape_reduced_count()
        )[0]

        # 2^Window x Window x Channels
        full_table: torch.FloatTensor = self(
            bin_in, dtype=torch.float, interleave=False, table=False
        )
        main_table = full_table[:, output_ind]  # 2^Window x Channels
        # Restructure the front delay table
        # 2x...(window times) x Window x Channels
        reshaped_full_table = full_table.reshape(
            *([2] * self.window), self.window, full_table.shape[-1]
        )
        unmasked_front_table = reshaped_full_table[..., :output_ind, :]
        front_mask = torch.ones(
            unmasked_front_table.shape,
            dtype=torch.bool,
            device=self.device_manager.device,
        )
        for num_drop_end in range(1, output_ind + 1):
            # As we go back, we want to drop the last i bits
            slicer = dynamic_slice(
                front_mask,
                dim=[self.window - j - 1 for j in range(num_drop_end)] + [self.window],
                index=([0] * num_drop_end) + [output_ind - num_drop_end],
            )
            front_mask[slicer] = False
        # 2^Window x FrontBits x Channels masked tensor
        front_table = MaskedTensor(
            tensor=unmasked_front_table, mask=front_mask, no_fill=True
        ).reshape(2**self.window, output_ind, full_table.shape[-1])

        ## DEBUG - Something funky is going on here. Numbers that should be the same
        ## in the output table, are almost the same, but off by ~1e-6: Running the
        ## below code will show this. This seems to be specific to the GPU, running
        ## on a CPU will get the expected results.
        # ind = 1
        # bin_in = enumerate_binary_inputs(self.window, device=self.device_manager.device)
        # for b, val, m in zip(
        #     bin_in,
        #     front_table.tensor[:, ind, 0],
        #     front_table.mask[:, ind, 0],
        # ):
        #     print(
        #         f"{''.join([str(bi.item()) for bi in b])} : {val.item()} : {m.item()}"
        #     )

        # Get the end delay tables
        # (2x...(window times) x Channels)
        unmasked_end_table = reshaped_full_table[..., output_ind + 1 :, :]
        end_mask = torch.ones(
            unmasked_end_table.shape,
            dtype=torch.bool,
            device=self.device_manager.device,
        )
        for num_drop_front in range(1, self.window - output_ind):
            # As we go forward, we want to drop the first i+1 bits
            slicer = dynamic_slice(
                end_mask,
                dim=[j for j in range(num_drop_front)] + [self.window],
                index=([0] * num_drop_front) + [num_drop_front - 1],
            )
            # print(slicer)
            # print(end_mask.shape)
            end_mask[slicer] = False
        # 2^Window x EndBits x Channels masked tensor
        end_table = MaskedTensor(
            tensor=unmasked_end_table, mask=end_mask, no_fill=True
        ).reshape(2**self.window, self.window - output_ind - 1, full_table.shape[-1])

        return {
            "front": front_table,
            "main": main_table,
            "end": end_table,
        }

    def to_table(self, constrain=True) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if not self.first_pad:
            print(
                "First pad is false, please note conv network will have different boundary codes."
            )

        table = self.to_full_table()["main"]

        if constrain:
            table = GeneralizedConvolutionalEncoder.opt_unit_power_constraint(
                table=table,
                delay=self.delay,
                window=self.window,
                input_size=self.input_size,
                multi=False,
                shift=True,
            )
        # Split into noninterleaved and interleaved
        table_noni = table[:, ..., :2]
        table_i = table[:, ..., 2:]

        return table_noni, table_i

    def compute_chi_values(
        self,
        received_symbols: torch.Tensor,
        channel: NoisyChannel,
        modulator: Modulator,
    ) -> Tuple[MaskedTensor, MaskedTensor]:
        batch_size, timesteps, channels = received_symbols.shape
        # We need to do this for boundary bits and the main body code
        # Boundary tables
        table_dict = self.to_full_table()
        front_table, main_table, end_table = (
            table_dict["front"],
            table_dict["main"],
            table_dict["end"],
        )

        # Compute evidence for factors of type 1. This comes from code table
        # Batch x Time x 2 x ... (Window times)
        # Front
        # 2^Window x FrontBits x Channels -> FrontBits x 2^Window x Channels
        front_table_t = front_table.transpose(0, 1)
        chi_values_front_unmasked = channel.log_prob(
            received_symbols[
                :, : front_table_t.shape[0], None
            ],  # B x FrontBits x 1 x C
            modulator.modulate(front_table_t.tensor)[
                None
            ],  # 1 x FrontBits x 2^Window x C
        )
        # B x FrontBits x 2^Window x C
        chi_values_front = MaskedTensor(
            tensor=chi_values_front_unmasked,
            mask=front_table_t.mask.broadcast_to(batch_size, *front_table_t.mask.shape),
            no_fill=True,
        )
        ## DEBUG - See note at line 404. Those effects are propagating down.
        # bin_in = enumerate_binary_inputs(self.window, device=self.device_manager.device)
        # for ind in range(chi_values_front.shape[1]):
        #     print(f"{ind} : ==================")
        #     for b, val, m in zip(
        #         bin_in,
        #         chi_values_front.tensor[0, ind, :, 0],
        #         chi_values_front.mask[0, ind, :, 0],
        #     ):
        #         print(
        #             f"{''.join([str(bi.item()) for bi in b])} : {val.item()} : {m.item()}"
        #         )

        # End
        # 2^Window x EndBits x Channels -> EndBits x 2^Window x Channels
        end_table_t = end_table.transpose(0, 1)
        chi_values_end_unmasked = channel.log_prob(
            received_symbols[:, -end_table_t.shape[0] :, None],  # B x EndBits x 1 x C
            modulator.modulate(end_table_t.tensor)[None],  # 1 x EndBits x 2^Window x C
        )
        # B x EndBits x 2^Window x C
        chi_values_end = MaskedTensor(
            tensor=chi_values_end_unmasked,
            mask=end_table_t.mask.broadcast_to(batch_size, *end_table_t.mask.shape),
            no_fill=True,
        )

        ## DEBUG - See note at line 404. Those effects are propagating down.
        # bin_in = enumerate_binary_inputs(self.window, device=self.device_manager.device)
        # for ind in range(chi_values_end.shape[1]):
        #     print(f"{ind} : ==================")
        #     for b, val, m in zip(
        #         bin_in,
        #         chi_values_end.tensor[0, ind, :, 0],
        #         chi_values_end.mask[0, ind, :, 0],
        #     ):
        #         print(
        #             f"{''.join([str(bi.item()) for bi in b])} : {val.item()} : {m.item()}"
        #         )

        # Main
        # B x T - FrontBits - EndBits x 2^W x C
        chi_values_main = channel.log_prob(
            received_symbols[
                :, chi_values_front.shape[1] : -chi_values_end.shape[1], None
            ],  # B x T - FrontBits - EndBits x 1 x C
            modulator.modulate(main_table)[None, None],  # 1 x 1 x 2^W x C
        )

        # Put them together
        # B x T x 2^W x C
        chi_values_unreduced: MaskedTensor = MaskedTensor.cat(
            [chi_values_front, chi_values_main, chi_values_end], dim=1
        )

        chi_values_noni: MaskedTensor = chi_values_unreduced[..., :2].sum(dim=-1)
        chi_values_i: MaskedTensor = chi_values_unreduced[..., 2:].sum(dim=-1)

        return (
            chi_values_noni,
            chi_values_i,
        )  # non-interleaved, interleaved

    def collect_evidence_from_chi_values(
        self,
        chi_values: MaskedTensor,
        input_symbols: list[str],
        received_factor_symbol: str = "y",
    ):
        evidence: Dict[str, NamedTensor] = {}
        batch_size = chi_values.shape[0]
        timesteps = chi_values.shape[1]
        chi_values = chi_values.reshape(batch_size, timesteps, *([2] * self.window))
        for i in range(timesteps):
            factor_i = f"{received_factor_symbol}_{i}"
            low = i - self.window + 1 + self.delay
            high = i + 1 + self.delay
            # Assumption is that indices outside our bounds are taken to be 0 input
            # This is synchronized with the creation of the masked tensors
            # slice_tuple = (
            #     [slice(None), i]  # Batch, Time
            #     + [0 for _ in range(low, 0)]
            #     + [slice(None) for _ in range(max(0, low), min(high, timesteps))]
            #     + [0 for _ in range(timesteps, high)]
            # )
            # If True, the input size is smaller than the window, and I don't
            # think things will work
            assert not (low < 0 and high > timesteps)
            slice_tuple = (
                [slice(None), i]  # Batch, Time
                # This will remove any bits that the end doesn't depend on
                + [0 for _ in range(timesteps, high)]
                + [slice(None) for _ in range(max(0, low), min(high, timesteps))]
                # This will remove any bits that the front doesn't depend on
                + [0 for _ in range(low, 0)]
            )
            # Here we should only have selected chi_values that are unmasked
            selected_chi_values: MaskedTensor = chi_values[slice_tuple]
            # print(selected_chi_values.tensor)
            # print(selected_chi_values.mask)
            # print(i)
            # print(selected_chi_values.shape)

            assert not torch.any(selected_chi_values.mask)
            evidence[factor_i] = NamedTensor(
                selected_chi_values.tensor,  # Assertion above ensures this is a well-defined tensor
                dim_names=[
                    input_symbols[j] for j in range(max(0, low), min(high, timesteps))
                ],
                batch_dims=1,
            )

        factor_groups, _ = nonrecursive_convolutional_factors(
            self.input_size,
            self.window,
            delay=self.delay,
            input_symbols=input_symbols,
            factor_symbol=received_factor_symbol,
        )

        assert {
            factor: evidence_data._dim_name_set
            for factor, evidence_data in evidence.items()
        } == factor_groups

        return evidence

    def compute_evidence(
        self,
        received_symbols: torch.Tensor,
        channel: NoisyChannel,
        modulator: Modulator,
        input_symbols: list[str] = None,
        received_factor_symbol: str = "y",
        prime_symbol: str = "p",
    ) -> Dict[str, NamedTensor]:
        # In this case, we can just fall back to our usual way of doing things
        if self.first_pad:
            return self.to_conv_code(constrain=True).compute_evidence(
                received_symbols=received_symbols,
                channel=channel,
                modulator=modulator,
                input_symbols=input_symbols,
                received_factor_symbol=received_factor_symbol,
            )

        # Otherwise we need to handle boundary codes.
        batch_size, timesteps, channels = received_symbols.shape
        chi_values_noni, chi_values_i = self.compute_chi_values(
            received_symbols=received_symbols, channel=channel, modulator=modulator
        )

        if input_symbols is None:
            input_symbols = [f"{INPUT_SYMBOL}_{i}" for i in range(timesteps)]

        interleaved_inputs = [
            input_symbols[self.interleaver.permutation[i]] for i in range(timesteps)
        ]

        noninterleaved_evidence = self.collect_evidence_from_chi_values(
            chi_values_noni,
            input_symbols=input_symbols,
            received_factor_symbol=received_factor_symbol,
        )
        interleaved_evidence = self.collect_evidence_from_chi_values(
            chi_values_i,
            input_symbols=interleaved_inputs,
            received_factor_symbol=received_factor_symbol + prime_symbol,
        )

        return {**noninterleaved_evidence, **interleaved_evidence}

    def settings(self) -> Dict[str, Any]:
        return {
            **super().settings(),
            "enc_num_layer": self.enc_num_layer.item(),
            "code_rate_k": self.code_rate_k.item(),
            "enc_num_unit": self.enc_num_unit.item(),
            "enc_kernel_size": self.enc_kernel_size.item(),
            "interleaver": self.interleaver.settings(),
            "type": self.__class__.__name__,
            **(
                {"mean": self.mean, "std": self.std}
                if self.mean_std_initialized
                else {}
            ),
        }

    def dependency_graph(self) -> InferenceGraph:
        return nonrecursive_turbo_graph(
            self.interleaver.permutation,
            self.window,
            delay=self.delay,
        )

    def long_settings(self) -> Dict[str, Any]:
        return {
            **super().long_settings(),
            "enc_num_layer": self.enc_num_layer.item(),
            "code_rate_k": self.code_rate_k.item(),
            "enc_num_unit": self.enc_num_unit.item(),
            "enc_kernel_size": self.enc_kernel_size.item(),
            "interleaver": self.interleaver.long_settings(),
            "type": self.__class__.__name__,
        }
