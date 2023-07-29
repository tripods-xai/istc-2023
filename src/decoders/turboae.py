import torch
import torch.nn.functional as F

from ..utils import DeviceManager, DEFAULT_DEVICE_MANAGER
from ..interleavers import Interleaver, TurboAEInterleaver
from ..neural_utils import SameShapeConv1d

from .decoder import SoftDecoder


class TurboAEDecoder(SoftDecoder):
    def __init__(
        self,
        num_iteration: int = 6,
        num_iter_ft: int = 5,
        dec_num_layer: int = 5,
        dec_num_unit: int = 100,
        dec_kernel_size: int = 5,
        front_pad: bool = False,
        interleaver: Interleaver = None,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ):
        super(TurboAEDecoder, self).__init__(device_manager=device_manager)

        self.num_iteration = num_iteration
        self.num_iter_ft = num_iter_ft
        self.dec_num_layer = dec_num_layer
        self.dec_num_unit = dec_num_unit
        self.dec_kernel_size = dec_kernel_size
        self.front_pad = front_pad

        # # XXX: This is a change in behavior!
        # pad_value = -1.0
        pad_value = 0.0

        if interleaver is None:
            interleaver = TurboAEInterleaver(device_manager=self.device_manager)

        self.interleaver = interleaver

        self.dec1_cnns = torch.nn.ModuleList()
        self.dec2_cnns = torch.nn.ModuleList()
        self.dec1_outputs = torch.nn.ModuleList()
        self.dec2_outputs = torch.nn.ModuleList()

        def CNNLayer(*args, **kwargs):
            return SameShapeConv1d(
                *args, front_pad=self.front_pad, pad_value=pad_value, **kwargs
            ).to(self.device_manager.device)

        for idx in range(self.num_iteration):
            self.dec1_cnns.append(
                CNNLayer(
                    num_layer=self.dec_num_layer,
                    in_channels=2 + self.num_iter_ft,
                    out_channels=self.dec_num_unit,
                    kernel_size=self.dec_kernel_size,
                )
            )

            self.dec2_cnns.append(
                CNNLayer(
                    num_layer=self.dec_num_layer,
                    in_channels=2 + self.num_iter_ft,
                    out_channels=self.dec_num_unit,
                    kernel_size=self.dec_kernel_size,
                )
            )
            self.dec1_outputs.append(
                torch.nn.Linear(self.dec_num_unit, self.num_iter_ft).to(
                    self.device_manager.device
                )
            )

            if idx == self.num_iteration - 1:
                self.dec2_outputs.append(
                    torch.nn.Linear(self.dec_num_unit, 1).to(self.device_manager.device)
                )
            else:
                self.dec2_outputs.append(
                    torch.nn.Linear(self.dec_num_unit, self.num_iter_ft).to(
                        self.device_manager.device
                    )
                )

        self.to(self.device_manager.device)

    # def input_pad(self, input_1, input_2, prior):
    #     F.pad(input_1, self.dec1_cnns[0].make_pad_tuple(self.dec_kernel_size - 1), value=-1.0)
    #     F.pad(input_2, self.dec1_cnns[0].make_pad_tuple(self.dec_kernel_size - 1), value=-1.0)
    #     F.pad(prior, self.dec1_cnns[0].make_pad_tuple(self.dec_kernel_size - 1), value=0.0)

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
        check_ks = [
            "_num_iteration",
            "_num_iter_ft",
            "_dec_num_layer",
            "_dec_num_unit",
            "_dec_kernel_size",
        ]
        for k in [k for k in check_ks if k in state_dict]:
            expected = state_dict.pop(k).item()
            actual = getattr(self, k[1:])
            assert (
                actual == expected
            ), f"Check {k} failed: self.{k[1:]}={actual} but state_dict[{k}]={expected}"
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

    def settings(self):
        return {
            "num_iteration": self.num_iteration,
            "num_iter_ft": self.num_iter_ft,
            "dec_num_layer": self.dec_num_layer,
            "dec_num_unit": self.dec_num_unit,
            "dec_kernel_size": self.dec_kernel_size,
            "interleaver": self.interleaver.settings(),
        }

    def long_settings(self):
        return {
            "num_iteration": self.num_iteration,
            "num_iter_ft": self.num_iter_ft,
            "dec_num_layer": self.dec_num_layer,
            "dec_num_unit": self.dec_num_unit,
            "dec_kernel_size": self.dec_kernel_size,
            "interleaver": self.interleaver.long_settings(),
        }

    @property
    def source_data_len(self):
        return len(self.interleaver)

    def set_parallel(self):
        for idx in range(self.num_iteration):
            self.dec1_cnns[idx] = torch.nn.DataParallel(self.dec1_cnns[idx])
            self.dec2_cnns[idx] = torch.nn.DataParallel(self.dec2_cnns[idx])
            self.dec1_outputs[idx] = torch.nn.DataParallel(self.dec1_outputs[idx])
            self.dec2_outputs[idx] = torch.nn.DataParallel(self.dec2_outputs[idx])

    def forward(self, received):

        received = received.float().to(self.device_manager.device)
        r_sys = received[:, :, 0:1]
        r_sys_int = self.interleaver.interleave(r_sys)
        r_par1 = received[:, :, 1:2]
        r_par2 = received[:, :, 2:3]

        prior = torch.zeros(
            (received.shape[0], received.shape[1], self.num_iter_ft),
            device=self.device_manager.device,
        )

        for idx in range(self.num_iteration - 1):
            x_this_dec = torch.cat([r_sys, r_par1, prior], dim=2)

            x_dec = self.dec1_cnns[idx](x_this_dec)
            x_plr = self.dec1_outputs[idx](x_dec)

            x_plr = x_plr - prior

            x_plr_int = self.interleaver.interleave(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int], dim=2)

            x_dec = self.dec2_cnns[idx](x_this_dec)

            x_plr = self.dec2_outputs[idx](x_dec)

            x_plr = x_plr - x_plr_int

            prior = self.interleaver.deinterleave(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys, r_par1, prior], dim=2)

        x_dec = self.dec1_cnns[self.num_iteration - 1](x_this_dec)
        x_plr = self.dec1_outputs[self.num_iteration - 1](x_dec)

        x_plr = x_plr - prior

        x_plr_int = self.interleaver.interleave(x_plr)

        x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int], dim=2)

        x_dec = self.dec2_cnns[self.num_iteration - 1](x_this_dec)
        x_plr = self.dec2_outputs[self.num_iteration - 1](x_dec)

        # We return the logits instead
        final = self.interleaver.deinterleave(x_plr)

        return final[:, :, 0]
