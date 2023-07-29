import argparse
import os

from src.utils import DeviceManager, get_timestamp
from src.channels import AWGN
from src.encoders import get_encoder
from src.modulation import BPSK
from src.measurements import TrellisConditionalEntropySampler
from src.engine import TqdmProgressBar, FileLogger, ResultsProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--encoder", required=True)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--block_len", type=int, default=10)
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--snr", type=float, default=0.0)
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--logdir", type=str, default="../logs")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    manager = DeviceManager(no_cuda=args.no_cuda, seed=args.seed)

    encoder = get_encoder(args.encoder)(
        num_steps=args.block_len, device_manager=manager
    )
    modulator = BPSK(device_manager=manager)
    channel = AWGN(snr=args.snr, device_manager=manager)
    cond_entropy_estimator = TrellisConditionalEntropySampler(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        device_manager=manager,
    )

    file_logger = FileLogger(
        preamble={
            "args": vars(args),
            "encoder_name": args.encoder,
            "encoder": encoder.long_settings(),
            "modulator": modulator.long_settings(),
            "channel": channel.long_settings(),
        },
        fp=os.path.join(
            args.logdir, f"{os.path.basename(__file__)}.{get_timestamp()}.json"
        ),
    )
    print(f"Logging results to {file_logger.fp}")
    num_batches = args.num_samples // args.batch_size
    progress_bar = TqdmProgressBar(total=num_batches)
    results_processor = ResultsProcessor(listeners=[progress_bar, file_logger])

    for i in range(0, args.num_samples, args.batch_size):
        cur_batch_size = min(args.num_samples - i, args.batch_size)
        sampled_cond_entropies = cond_entropy_estimator.sample(cur_batch_size)
        records = {
            "ce": sampled_cond_entropies,
        }
        results_processor.update(records)

    progress_bar.close()
