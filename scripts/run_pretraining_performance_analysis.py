import argparse
import json
import pathlib
from typing import Dict, Tuple

import pandas as pd
from tabulate import tabulate

from plausibility_vaccine.util.path import get_root_dir

parser = argparse.ArgumentParser(
    description='Aggregate pretrained adapter performance results',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--results-dir',
    type=str,
    default='results/albert_reduce_factor_64',
    help='Path to root directory containing results',
)
parser.add_argument(
    '--artifacts-dir',
    type=str,
    default='artifacts/performance/pretraining',
    help='Path to artifact directory containing plots',
)


def main() -> None:
    args = parser.parse_args()
    results_dir = get_root_dir() / args.results_dir
    if not results_dir.is_dir():
        raise FileNotFoundError(f'Results directory: {results_dir.resolve()}')

    property_adapters_perf, verb_adapters_perf = parse_adapter_results(results_dir)
    property_tables = generate_property_perf_tables(property_adapters_perf)
    verb_tables = generate_verb_perf_tables(verb_adapters_perf)

    latex_tables = property_tables | verb_tables  # type: ignore
    save_latex_tables(latex_tables, args.artifacts_dir)


def generate_property_perf_tables(property_perf: pd.DataFrame) -> Dict[str, str]:
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    keep_cols = ['task']
    for c in property_perf.columns:
        if any([metric in c for metric in metrics]):
            keep_cols.append(c)

    property_perf = pd.melt(property_perf[keep_cols], 'task')
    property_perf['metric'] = property_perf['variable'].apply(
        lambda x: x.split('_')[-1]
    )
    property_perf['shard'] = property_perf['variable'].apply(
        lambda x: x.split('_')[1].split('-')[-1]
    )
    property_perf['task'] = property_perf['task'].apply(lambda x: x.split('_')[0])
    property_perf = property_perf.drop('variable', axis=1)

    mu = property_perf.groupby(['task', 'metric'])['value'].mean().reset_index()
    std = property_perf.groupby(['task', 'metric'])['value'].std().reset_index()

    mu = mu.rename({'value': 'mu'}, axis=1)
    std = std.rename({'value': 'std'}, axis=1)

    data = pd.merge(mu, std, how='left', on=['task', 'metric'])
    data['value'] = data.apply(lambda x: f'{x["mu"]:.2} ± {x["std"]:.2}', axis=1)
    data = data.drop(['mu', 'std'], axis=1)
    table_data = data.pivot_table(
        values=['value'], index=['task'], columns='metric', aggfunc='first'
    )

    headers = ['Task'] + [metric[0].upper() + metric[1:] for metric in metrics]
    print(tabulate(table_data, headers, tablefmt='fancy_grid'))
    return {'property_adapters': tabulate(table_data, headers, tablefmt='latex')}


def generate_verb_perf_tables(verb_perf: pd.DataFrame) -> Dict[str, str]:
    metrics = ['mse']
    keep_cols = ['task']
    for c in verb_perf.columns:
        if any([metric in c for metric in metrics]):
            keep_cols.append(c)

    verb_perf = pd.melt(verb_perf[keep_cols], 'task')
    verb_perf['metric'] = verb_perf['variable'].apply(lambda x: x.split('_')[-1])
    verb_perf['shard'] = verb_perf['variable'].apply(
        lambda x: x.split('_')[1].split('-')[-1]
    )
    verb_perf = verb_perf.drop('variable', axis=1)

    mu = verb_perf.groupby(['task', 'metric'])['value'].mean().reset_index()
    std = verb_perf.groupby(['task', 'metric'])['value'].std().reset_index()

    mu = mu.rename({'value': 'mu'}, axis=1)
    std = std.rename({'value': 'std'}, axis=1)

    data = pd.merge(mu, std, how='left', on=['task', 'metric'])
    data['value'] = data.apply(lambda x: f'{x["mu"]:.2} ± {x["std"]:.2}', axis=1)
    data = data.drop(['mu', 'std'], axis=1)
    table_data = data.pivot_table(
        values=['value'], index=['task'], columns='metric', aggfunc='first'
    )

    headers = ['Task'] + [metric[0].upper() + metric[1:] for metric in metrics]
    print(tabulate(table_data, headers, tablefmt='fancy_grid'))
    return {'verb_adapters': tabulate(table_data, headers, tablefmt='latex')}


def save_latex_tables(latex_tables: Dict[str, str], artifacts_dir_str: str) -> None:
    artifacts_dir = pathlib.Path(artifacts_dir_str)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for table_name, table_latex in latex_tables.items():
        with open(artifacts_dir / f'{table_name}_performance.txt', 'w') as f:
            f.write(table_latex)


def parse_adapter_results(
    results_dir: pathlib.Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    property_adapters_dfs, verb_adapters_dfs = [], []
    for result_dir in results_dir.iterdir():
        if result_dir.is_dir():
            with open(result_dir / 'eval_results.json') as f:
                result_data = json.load(f)

            result_df = pd.DataFrame.from_dict(result_data, orient='index').T
            result_df['task'] = result_dir.name
            if 'verb' in result_dir.name:
                verb_adapters_dfs.append(result_df)
            elif 'pred' in result_dir.name:
                property_adapters_dfs.append(result_df)

    if not len(property_adapters_dfs):
        raise ValueError('No property adapter pretraining results found!')

    if not len(verb_adapters_dfs):
        raise ValueError('No verb adapter pretraining results found!')

    property_perf = pd.concat(property_adapters_dfs).sort_values('task')
    verb_perf = pd.concat(verb_adapters_dfs).sort_values('task')
    return property_perf, verb_perf


if __name__ == '__main__':
    main()
