import argparse
import pathlib
from typing import Dict

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

parser.add_argument(
    '--save-fig-dir',
    type=str,
    default='../figs/adapter_attention_weights/',
    help='Path to saved figure folder.',
)
parser.add_argument(
    '--chart-fig-dir',
    type=str,
    default='../figs/adapter_attention_weights/charts_attention_weights/',
    help='Path to saved attention weight charts.',
)
parser.add_argument(
    '--hist-fig-dir',
    type=str,
    default='../figs/adapter_attention_weights/hist_combined_weights/',
    help='Path to saved attention weight charts.',
)


def plot_weight_distributions(
    adapter_dict: Dict, task_name: str, save_path: pathlib.Path
) -> None:
    for name, _ in adapter_dict.items():
        if 'weight' in name:  # Ignore bias for now
            weights_np = adapter_dict[name].detach().cpu().numpy()
            if weights_np.shape == (len(weights_np),):
                weights_np = weights_np.reshape(
                    (len(weights_np), 1)
                )  # Column array needed for image format
            print(f'{task_name}')
            plt.figure(figsize=(10, 6))
            plt.imshow(weights_np, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(f'Dense Layer Weights: {task_name}')
            plt.savefig(save_path)
            plt.close()


def plot_combined_weight_hist_distributions(
    adapter_dict: Dict, task_name: str, save_path: pathlib.Path
) -> None:
    plt.figure(figsize=(12, 6))
    all_weights = []
    for name, _ in adapter_dict.items():
        weights_np = adapter_dict[name].detach().cpu().numpy().flatten()
        all_weights.extend(weights_np)

    plt.hist(all_weights, bins=100, alpha=0.5, label=f'{task_name}', density=True)
    plt.title(f'Combined Weight Distributions: {task_name}')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    args = parser.parse_args()
    config_file_path = get_root_dir() / args.config_file
    _, _, _, finetuning_args = parse_args(config_file_path)
    for task_name, _ in finetuning_args.pretraining_tasks.items():
        weight_path = (
            pathlib.Path(args.weights_dir) / f'{task_name}' / 'pytorch_model_head.bin'
        )
        save_path_charts = (
            pathlib.Path(args.chart_fig_dir) / f'{task_name}_attention_weights.png'
        )
        save_path_hist = (
            pathlib.Path(args.hist_fig_dir) / f'{task_name}_combined_weights.png'
        )
        adapter_dict = torch.load(
            weight_path, map_location=torch.device('cpu'), weights_only=True
        )
        plot_weight_distributions(adapter_dict, task_name, save_path_charts)

        plot_combined_weight_hist_distributions(adapter_dict, task_name, save_path_hist)


if __name__ == '__main__':
    main()
