import argparse
import json
import pathlib
from typing import Dict, List

import pandas as pd
from tabulate import tabulate

from plausibility_vaccine.util.path import get_root_dir

parser = argparse.ArgumentParser(
    description='Run ablation against various adapters',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--results-dir',
    type=str,
    default='results/',
    help='Path to root directory containing results',
)
parser.add_argument(
    '--ablation-names',
    type=List[str],
    default=[
        'albert_reduce_factor_64',
        'albert_reduce_factor_32',
        'albert_reduce_factor_16',
    ],
    help='List of adapter ablation',
)
parser.add_argument(
    '--artifacts-dir',
    type=str,
    default='artifacts/performance/ablations',
    help='Path to artifact directory containing plots',
)


def main() -> None:
    args = parser.parse_args()

    perf_dfs = []
    for ablation_name in args.ablation_names:
        results_dir = get_root_dir() / args.results_dir / ablation_name
        if not results_dir.is_dir():
            raise FileNotFoundError(f'Results directory: {results_dir.resolve()}')

        perf_df = parse_downstream_results(results_dir)
        perf_df['ablation_name'] = ablation_name
        perf_dfs.append(perf_df)

    perf = pd.concat(perf_dfs)

    latex_tables = generate_perf_tables(perf)
    save_latex_tables(latex_tables, args.artifacts_dir)


def generate_perf_tables(perf_df: pd.DataFrame) -> Dict[str, str]:
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    keep_cols = ['ablation_name', 'task']
    for c in perf_df.columns:
        if any([metric in c for metric in metrics]):
            keep_cols.append(c)

    perf_df = pd.melt(perf_df[keep_cols], ['ablation_name', 'task'])
    perf_df['metric'] = perf_df['variable'].apply(lambda x: x.split('_')[-1])
    perf_df['train_data'] = perf_df['task'].apply(
        lambda x: 'Combined' if 'combined' in x else 'Individual'
    )
    perf_df['shard'] = perf_df['variable'].apply(
        lambda x: x.split('_')[1].split('-')[-1]
    )
    perf_df['finetune_plausibility'] = perf_df['task'].apply(
        lambda x: 'MLP' if 'mlp' in x else 'Adapter'
    )
    perf_df['adapter_fusion'] = perf_df['task'].apply(
        lambda x: 'No Adapters'
        if 'base' in x
        else 'Property Adapters'
        if 'property' in x
        else 'Property + Verb Adapters'
    )
    perf_df['task'] = perf_df['task'].apply(lambda x: x.split('_')[0])
    perf_df = perf_df.drop('variable', axis=1)

    group_cols = [
        'ablation_name',
        'task',
        'adapter_fusion',
        'train_data',
        'finetune_plausibility',
        'metric',
    ]
    mu = perf_df.groupby(group_cols)['value'].mean().reset_index()
    std = perf_df.groupby(group_cols)['value'].std().reset_index()

    mu = mu.rename({'value': 'mu'}, axis=1)
    std = std.rename({'value': 'std'}, axis=1)

    data = pd.merge(mu, std, how='left', on=group_cols)
    data['value'] = data.apply(lambda x: f'{x["mu"]:.2} ± {x["std"]:.2}', axis=1)
    data = data.drop(['mu', 'std'], axis=1)
    table_data = data.pivot_table(
        values=['value'],
        index=[
            'ablation_name',
            'task',
            'train_data',
            'adapter_fusion',
            'finetune_plausibility',
        ],
        columns='metric',
        aggfunc='first',
    ).reset_index()

    headers = [
        'Ablation',
        'Task',
        'Training Data',
        'Adapter Fusion',
        'FineTune Plausibility',
    ] + [metric[0].upper() + metric[1:] for metric in metrics]
    print(tabulate(table_data, headers, tablefmt='fancy_grid'))
    return {'adapter_ablation': tabulate(table_data, headers, tablefmt='latex')}


def save_latex_tables(latex_tables: Dict[str, str], artifacts_dir_str: str) -> None:
    artifacts_dir = pathlib.Path(artifacts_dir_str)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for table_name, table_latex in latex_tables.items():
        with open(artifacts_dir / f'{table_name}.txt', 'w') as f:
            f.write(table_latex)


def parse_downstream_results(results_dir: pathlib.Path) -> pd.DataFrame:
    pep_dfs, twenty_dfs = [], []
    for result_dir in results_dir.iterdir():
        if result_dir.is_dir():
            with open(result_dir / 'eval_results.json') as f:
                result_data = json.load(f)

            result_df = pd.DataFrame.from_dict(result_data, orient='index').T
            result_df['task'] = result_dir.name
            if 'pep' in result_dir.name:
                pep_dfs.append(result_df)
            elif '20q' in result_dir.name:
                twenty_dfs.append(result_df)

    if not len(pep_dfs):
        raise ValueError('No pep plausibility results found!')

    if not len(twenty_dfs):
        raise ValueError('No 20q plausibility results found!')

    return pd.concat(pep_dfs + twenty_dfs).sort_values('task')


if __name__ == '__main__':
    main()
