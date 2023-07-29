import argparse
from pathlib import Path

from src.constants import OUTPUTS_DIR, EXPERIMENT_SETTINGS_JSON
from src.utils import tmp_if_debug

from src.experiments import run_experiments, run_all_experiments

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_id",
    help="The ID for the experiment to run. Not passing one will run all experiments.",
)
parser.add_argument(
    "--experiment_settings_path",
    type=Path,
    default=EXPERIMENT_SETTINGS_JSON,
    help="The path to the json file with experiment settings.",
)
parser.add_argument(
    "--log_every",
    type=int,
    default=200,
    help="Number of write to fiel logger before flushing.",
)
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--output_dir", type=Path, default=OUTPUTS_DIR)
parser.add_argument("--no_outputs", action="store_true")
parser.add_argument("--no_logging", action="store_true")
parser.add_argument("--debug", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    output_dir = (
        None if args.no_outputs else tmp_if_debug(args.output_dir, debug=args.debug)
    )

    if args.experiment_id is None:
        run_all_experiments(
            experiment_settings_json=args.experiment_settings_path,
            no_cuda=args.no_cuda,
            output_dir=output_dir,
            log_every=args.log_every,
        )
    else:
        run_experiments(
            args.experiment_id,
            experiment_settings_json=args.experiment_settings_path,
            no_cuda=args.no_cuda,
            output_dir=output_dir,
            log_every=args.log_every,
        )
