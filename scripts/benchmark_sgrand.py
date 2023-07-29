import argparse
import os

import torch

from src.utils import DeviceManager, get_timestamp
from src.channels import AWGN
from src.decoders import SGRAND
from src.encoders import ParityEncoder
from src.engine import TqdmProgressBar, FileLogger, ResultsProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--block_size", type=int, default=25)
parser.add_argument("--num_batches", type=int, default=100)
parser.add_argument("--rate", type=float, default=2 / 3)
parser.add_argument("--snr", type=float, default=3.0)
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--logdir", type=str, default="../logs")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    manager = DeviceManager(no_cuda=args.no_cuda, seed=args.seed)

    encoder = ParityEncoder(
        input_size=args.block_size,
        output_size=int(args.block_size / args.rate),
        device_manager=manager,
    )
    channel = AWGN(snr=args.snr, device_manager=manager)
    decoder = SGRAND(
        source_data_len=args.block_size,
        channel=channel,
        encoder=encoder,
        device_manager=manager,
    )

    file_logger = FileLogger(
        preamble={"args": vars(args)},
        fp=os.path.join(
            args.logdir, f"{os.path.basename(__file__)}.{get_timestamp()}.json"
        ),
    )
    print(f"Logging results to {file_logger.fp}")
    progress_bar = TqdmProgressBar(total=args.num_batches)
    results_processor = ResultsProcessor(listeners=[progress_bar, file_logger])

    for _ in range(args.num_batches):
        input_data = torch.randint(
            0,
            2,
            (args.batch_size, args.block_size),
            generator=manager.generator,
            device=manager.device,
        ).float()

        x = encoder(input_data)
        y = channel(x)
        decoded_data, log_ml, stats = decoder(y)

        bit_has_error = (decoded_data != input_data).long()
        block_has_error = torch.any(bit_has_error, dim=-1).long()

        records = {
            "ber": bit_has_error,
            "bler": block_has_error,
            "queries": stats["queries"],
        }
        results_processor.update(records)

    progress_bar.close()
