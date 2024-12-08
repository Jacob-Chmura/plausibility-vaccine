import argparse
import pathlib

import matplotlib.pyplot as plt
import torch

from plausibility_vaccine.util.args import parse_args
from plausibility_vaccine.util.path import get_root_dir

parser = argparse.ArgumentParser(
    description='Run analysis of attention weights',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='config/base.yaml',
    help='Path to yaml configuration file for iterating tasks.',
)
parser.add_argument(
    '--weights-dir',
    type=str,
    default='../weights/',
    help='Path to binary weight file.',
)


def main() -> None:
    args = parser.parse_args()
    config_file_path = get_root_dir() / args.config_file
    _, _, _, finetuning_args = parse_args(config_file_path)
    for task_name, _ in finetuning_args.pretraining_tasks.items():
        weight_path = (
            pathlib.Path(args.weights_dir) / f'{task_name}' / 'pytorch_adapter.bin'
        )
        state_dict = torch.load(
            weight_path, map_location=torch.device('cpu'), weights_only=True
        )

        for name, param in state_dict.items():
            if 'weight' in name:  # Ignore bias for now
                print(f'{name}: {param.shape}')
                weights_np = state_dict[name].detach().cpu().numpy()
                plt.figure(figsize=(10, 6))
                plt.imshow(weights_np, cmap='viridis', aspect='auto')
                plt.colorbar()
                plt.title('Dense Layer Weights')
                plt.show()


if __name__ == '__main__':
    main()
