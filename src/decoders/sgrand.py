from typing import Any, Tuple, NamedTuple, Dict

import torch

from .decoder import HardDecoder

from ..data_structures import SimplePQ, TensorStack
from ..channels import NoisyChannel
from ..utils import DeviceManager, DEFAULT_DEVICE_MANAGER, get_dummy
from ..encoders import Encoder


class SGRANDOutput(NamedTuple):
    decoded_data: torch.FloatTensor
    log_ml: torch.FloatTensor
    stats: Dict[str, Any]


class SGRAND(HardDecoder):
    def __init__(
        self,
        source_data_len: int,
        channel: NoisyChannel,
        encoder: Encoder,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ) -> None:
        super().__init__(device_manager)
        self._source_data_len = source_data_len
        self.channel = channel
        self.encoder = encoder

    @property
    def source_data_len(self):
        return self._source_data_len

    def get_children(
        self,
        error_vectors_oei: torch.BoolTensor,
    ) -> Tuple[
        Tuple[torch.BoolTensor, torch.BoolTensor],
        Tuple[torch.BoolTensor, torch.BoolTensor],
    ]:
        batch_size = error_vectors_oei.shape[0]
        data_len = error_vectors_oei.shape[1]
        child1 = error_vectors_oei.clone()
        has_child1 = torch.zeros(batch_size, dtype=torch.bool)
        has_child2 = torch.zeros(batch_size, dtype=torch.bool)

        index_active = torch.where(
            error_vectors_oei, get_dummy(error_vectors_oei.shape, 1), -1
        )
        j_star, _ = torch.max(index_active, dim=1)  # -1 if e is all 0s
        # print(f"The error vectors {error_vectors_oei}")
        # print(f"The j_star {j_star}")
        # Child 1
        j_star_plus_1 = j_star + 1
        has_child1 |= j_star_plus_1 < data_len  # We have a bit to flip
        child1[has_child1, j_star_plus_1[has_child1]] = True

        # Child 2
        child2 = child1.clone()
        has_child2 |= has_child1 & (j_star > -1)
        child2[has_child2, j_star[has_child2]] = False

        return ((child1[has_child1], has_child1), (child2[has_child2], has_child2))

    def forward(self, y: torch.Tensor):
        """
        Args:
            y (???): The received noisy sequence.

        Shape:
            - y: (batch_size, block_len, ...)

        Returns:

        """
        batch_size = y.shape[0]
        y_flat = y.reshape((batch_size, -1))
        data_len = y_flat.shape[1]

        error_candidates_oei = SimplePQ(
            batch_size, data_shape=(data_len,), dtype=torch.bool
        )
        decoded_data = torch.zeros((batch_size, self.source_data_len))
        decoding_complete = torch.zeros((batch_size,), dtype=torch.bool)
        log_ml = torch.zeros((batch_size,))

        y_hat = self.channel.demodulate(y_flat)
        log_likelihood_y_hat = self.channel.log_prob(y_flat, y_hat)
        log_likelihood_not_y_hat = self.channel.log_prob(
            y_flat,
            self.channel.apply_error_vector(
                y_hat, torch.full_like(y_hat, True, dtype=torch.bool)
            ),
        )
        log_posterior = log_likelihood_y_hat - torch.logaddexp(
            log_likelihood_y_hat, log_likelihood_not_y_hat
        )

        oei = torch.argsort(log_posterior, dim=-1, descending=False)
        inverse_oei = torch.argsort(oei, dim=-1)

        # Initital candidate
        # error_candidates will be oei permuted, but this first insertion is same permuted
        # or unpermuted.
        error_candidates_oei.insert(
            keys=torch.sum(log_likelihood_y_hat, dim=-1),
            data=torch.zeros_like(y_hat, dtype=torch.bool),
        )
        batch_dummy = get_dummy(log_likelihood_y_hat.shape, dim=0)
        log_likelihood_y_hat_oei = log_likelihood_y_hat[batch_dummy, oei]
        log_likelihood_not_y_hat_oei = log_likelihood_not_y_hat[batch_dummy, oei]

        # For debugging
        # likelihood_trajectory = TensorStack(
        #     batch_size=batch_size, data_shape=(), dtype=torch.float32, init_size=10
        # )
        # error_vector_oei_trajectory = TensorStack(
        #     batch_size=batch_size,
        #     data_shape=(data_len,),
        #     dtype=torch.bool,
        #     init_size=10,
        # )
        query_count = torch.zeros((batch_size,), dtype=torch.int64)

        while not torch.all(decoding_complete):
            (
                this_likelihood,
                this_candidate_oei,
                this_batches,
            ) = error_candidates_oei.pop()
            selected_y_hat = y_hat[this_batches]

            debug_batch_mask = torch.zeros((batch_size,), dtype=torch.bool)
            debug_batch_mask[this_batches] = True
            # likelihood_trajectory.push(this_likelihood, batch_mask=debug_batch_mask)
            # error_vector_oei_trajectory.push(
            #     this_candidate_oei, batch_mask=debug_batch_mask
            # )

            # Check if we've reached a codeword
            this_candidate = this_candidate_oei[
                get_dummy(this_candidate_oei.shape, 0), inverse_oei[this_batches]
            ]
            is_codeword, decoding = self.encoder.is_codeword(
                self.channel.apply_error_vector(selected_y_hat, this_candidate)
            )
            query_count[this_batches] += 1
            
            complete_inds = this_batches[is_codeword]
            decoding_complete[complete_inds] = True
            decoded_data[complete_inds] = decoding[is_codeword]
            log_ml[complete_inds] = this_likelihood[is_codeword]
            if torch.all(decoding_complete):
                break

            # Remove the batches that are done
            # print(
            #     f"Dropped {torch.sum(is_codeword.long())} batches out of {is_codeword.shape[0]} batches"
            # )
            # print(
            #     f"Remaining OEI vectors that were popped: {this_candidate_oei.long()}"
            # )
            # print(f"Remaining likelihoods that were popped: {this_likelihood.long()}")
            # From here down we've removed all batches that are complete
            error_candidates_oei.drop_batches(is_codeword)
            this_likelihood = this_likelihood[~is_codeword]
            this_candidate_oei = this_candidate_oei[~is_codeword]
            this_batches = this_batches[~is_codeword]

            # Otherwise we need to keep looking
            (selected_child1, has_child1), (
                selected_child2,
                has_child2,
            ) = self.get_children(this_candidate_oei)

            # All batches without child1 are removed
            child1_log_likelihood = torch.sum(
                torch.where(
                    selected_child1,
                    log_likelihood_not_y_hat_oei[this_batches[has_child1]],
                    log_likelihood_y_hat_oei[this_batches[has_child1]],
                ),
                dim=1,
            )
            # print(f"Parent OEI for child 1: {this_candidate_oei[has_child1].long()}")
            # print(f"Child 1: {selected_child1.long()}")
            # All batches without child2 are removed
            child2_log_likelihood = torch.sum(
                torch.where(
                    selected_child2,
                    log_likelihood_not_y_hat_oei[this_batches[has_child2]],
                    log_likelihood_y_hat_oei[this_batches[has_child2]],
                ),
                dim=1,
            )
            # print(f"Parent OEI for child 2: {this_candidate_oei[has_child2].long()}")
            # print(f"Child 2: {selected_child2.long()}")
            # Insert only into batches with child1
            error_candidates_oei.insert(
                child1_log_likelihood, selected_child1, has_child1
            )
            # Insert only into batches with child2
            error_candidates_oei.insert(
                child2_log_likelihood, selected_child2, has_child2
            )
        # torch.set_printoptions(profile="full")
        # print(f"Likelihood trajectory {likelihood_trajectory.get_data()}")
        # print(
        #     f"Error Vector OEI trajectory {error_vector_oei_trajectory.get_data().long()}"
        # )
        return SGRANDOutput(decoded_data, log_ml, stats={"queries": query_count})
