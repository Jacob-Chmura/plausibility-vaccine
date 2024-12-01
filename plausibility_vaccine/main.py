import argparse

from plausibility_vaccine.run import run
from plausibility_vaccine.util.args import parse_args
from plausibility_vaccine.util.logging import setup_basic_logging
from plausibility_vaccine.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Plausibility Vaccine',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='config/base.yaml',
    help='Path to yaml configuration file to use',
)


def main() -> None:
    args = parser.parse_args()
    meta_args, model_args, training_args, finetuning_args = parse_args(args.config_file)
    setup_basic_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)
    run(model_args, training_args, finetuning_args)


if __name__ == '__main__':
    main()
