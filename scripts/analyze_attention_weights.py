import argparse
import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from adapters import AutoAdapterModel
from bertviz import head_view
from sklearn.decomposition import PCA
from transformers import AutoTokenizer

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
    default='weights/',
    help='Path to binary weight file.',
)

parser.add_argument(
    '--save-fig-dir',
    type=str,
    default='figs/adapter_attention_weights/',
    help='Path to saved figure folder.',
)
parser.add_argument(
    '--chart-fig-dir',
    type=str,
    default='figs/adapter_attention_weights/charts_attention_weights/',
    help='Path to saved attention weight charts.',
)
parser.add_argument(
    '--hist-fig-dir',
    type=str,
    default='figs/adapter_attention_weights/hist_combined_weights/',
    help='Path to saved attention weight histograms.',
)
parser.add_argument(
    '--pca-fig-dir',
    type=str,
    default='figs/adapter_attention_weights/pca_combined_weights/',
    help='Path to saved attention weight PCA charts.',
)
parser.add_argument(
    '--bertviz-fig-dir',
    type=str,
    default='figs/adapter_attention_weights/bert_viz/',
    help='Path to saved bert viz HTML.',
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


def plot_pca_l1(adapter_dict: Dict, task_name: str, save_path: pathlib.Path) -> None:
    weights_np = None
    for name, _ in adapter_dict.items():
        if 'bias' not in name:
            weights_np = adapter_dict[name].detach().cpu().numpy()
            print(weights_np)
            if weights_np.shape == (len(weights_np),):
                weights_np = weights_np.reshape(
                    (len(weights_np), 1)
                )  # Column array needed for image format
            print(f'{name} of shape : {weights_np.shape}')
            break

    pca = PCA(n_components=2)
    weights_pca = pca.fit_transform(weights_np)
    print(weights_pca)

    # Get principal directions
    mean = np.mean(weights_pca, axis=0)
    components = pca.components_
    explained_variances = pca.explained_variance_
    scaling_factor = 10

    plt.figure(figsize=(10, 6))
    plt.scatter(weights_pca[:, 0], weights_pca[:, 1], s=50)

    # Plot the PCA lines
    for i, (component, variance) in enumerate(zip(components, explained_variances)):
        line = scaling_factor * np.sqrt(variance) * component
        plt.plot(
            [mean[0] - line[0], mean[0] + line[0]],
            [mean[1] - line[1], mean[1] + line[1]],
            linestyle='--',
            linewidth=2,
            label=f'PC{i+1}',
        )
    plt.title(f'PCA of weights: {task_name}')
    plt.xlabel('PC_axis_1')
    plt.ylabel('PC_axis_2')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_bert_viz_display(
    adapter_path: pathlib.Path, task_name: str, output_path: pathlib.Path
) -> None:
    model_name = 'albert-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)

    model.load_adapter(str(adapter_path), load_as=task_name)
    model.set_active_adapters(task_name)

    sentence = 'monkey eats apple'
    inputs = tokenizer(sentence, return_tensors='pt')

    output = model(**inputs, output_attentions=True)
    attention = output.attentions
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    html_display = head_view(attention, tokens, html_action='return')
    with open(output_path, 'w') as f:
        f.write(html_display.data)


def main() -> None:
    args = parser.parse_args()
    config_file_path = get_root_dir() / args.config_file
    _, _, _, finetuning_args = parse_args(config_file_path)
    for task_name, _ in finetuning_args.pretraining_tasks.items():
        weight_path = (
            get_root_dir()
            / args.weights_dir
            / f'{task_name}'
            / 'pytorch_model_head.bin'
        )

        adapter_path = get_root_dir() / args.weights_dir / f'{task_name}'

        save_path_charts = (
            get_root_dir() / args.chart_fig_dir / f'{task_name}_attention_weights.png'
        )

        save_path_hist = (
            get_root_dir() / args.hist_fig_dir / f'{task_name}_combined_weights.png'
        )
        save_path_pca = (
            get_root_dir() / args.pca_fig_dir / f'{task_name}_attention_pca.png'
        )
        save_path_bertviz = (
            get_root_dir() / args.bertviz_fig_dir / f'{task_name}_attention_viz.html'
        )
        adapter_dict = torch.load(
            weight_path, map_location=torch.device('cpu'), weights_only=True
        )
        plot_weight_distributions(adapter_dict, task_name, save_path_charts)

        plot_combined_weight_hist_distributions(adapter_dict, task_name, save_path_hist)

        plot_pca_l1(adapter_dict, task_name, save_path_pca)

        get_bert_viz_display(
            adapter_path=adapter_path,
            task_name=task_name,
            output_path=save_path_bertviz,
        )
        break


if __name__ == '__main__':
    main()
