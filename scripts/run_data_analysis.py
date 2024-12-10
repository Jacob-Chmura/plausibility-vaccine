import argparse
import pathlib
from typing import Dict

import pandas as pd
from tabulate import tabulate

from plausibility_vaccine.util.path import get_root_dir

pd.options.display.max_rows = 100

parser = argparse.ArgumentParser(
    description='Run analysis of datasets',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--plausibility-datasets-dir',
    type=str,
    default='data/plausibility_data/',
    help='Path to root directory containing plausibility data',
)
parser.add_argument(
    '--property-datasets-dir',
    type=str,
    default='data/property_data/',
    help='Path to root directory containing property data',
)
parser.add_argument(
    '--verb-datasets-dir',
    type=str,
    default='data/verb_understanding_data/',
    help='Path to root directory containing verb data',
)
parser.add_argument(
    '--artifacts-dir',
    type=str,
    default='artifacts/dataset_statistics',
    help='Path to artifact directory containing plots',
)


def main() -> None:
    args = parser.parse_args()
    plausibility_tables = generate_plausibility_tables(args.plausibility_datasets_dir)
    property_tables = generate_property_tables(args.property_datasets_dir)
    verb_tables = generate_verb_tables(args.verb_datasets_dir)

    latex_tables = plausibility_tables | property_tables | verb_tables  # type: ignore
    save_latex_tables(latex_tables, args.artifacts_dir)


def generate_plausibility_tables(datasets_dir: str) -> Dict[str, str]:
    df = _read_subdirectory_dataset_csvs(datasets_dir)
    df = df.groupby(['task', 'split'])['label'].value_counts(normalize=True)
    table_data = df.round(3).reset_index()

    headers = ['Task', 'Split', 'Plausibility Label', 'Proportion']
    print(tabulate(table_data, headers, tablefmt='fancy_grid'))
    return {'plausibility_data': tabulate(table_data, headers, tablefmt='latex')}


def generate_property_tables(datasets_dir: str) -> Dict[str, str]:
    PROPERTY_LABEL_BINS = {
        'Mobility': ['rock', 'tree', 'mushroom', 'animal', 'man', 'car'],
        'Opacity': ['glass', 'paper', 'cloud', 'man', 'wall'],
        'Phase': ['smoke', 'milk', 'wood', 'diamond'],
        'Rigidity': ['water', 'skin', 'leather', 'wood', 'metal'],
        'Sentience': ['rock', 'tree', 'ant', 'cat', 'chimp', 'man'],
        'Shape': ['square', 'sphere', 'ant', 'man', 'cloud'],
        'Size': ['ant', 'cat', 'person', 'jeep', 'stadium'],
        'Temperature': ['ice', 'soup', 'fire', 'lava', 'sun'],
        'Texture': ['glass', 'carpet', 'book', 'ant', 'man', 'sandpaper'],
        'Weight': ['watch', 'book', 'man', 'jeep', 'stadium'],
    }

    tables = {}
    df = _read_subdirectory_dataset_csvs(datasets_dir)
    for task_name, df_task in df.groupby('task'):
        df_task = (
            df_task.groupby(['task'])['label']
            .value_counts(normalize=True)
            .reset_index()
            .sort_values('label')
        )
        df_task['label'] = df_task['label'].apply(
            lambda x: PROPERTY_LABEL_BINS[task_name][x - 1]
        )
        table_data = df_task.round(3).reset_index()[['task', 'label', 'proportion']]

        headers = ['Task', 'Bin Label', 'Proportion']
        print(tabulate(table_data, headers, tablefmt='fancy_grid'))
        tables[f'{task_name}_data'] = tabulate(table_data, headers, tablefmt='latex')
    return tables


def generate_verb_tables(datasets_dir: str) -> Dict[str, str]:
    df = _read_subdirectory_dataset_csvs(datasets_dir)
    percentiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

    tables = {}
    for task_name, df_task in df.groupby('task'):
        df_task = df_task.dropna(axis='columns')
        df_task = df_task.groupby('task')['label'].describe(percentiles=percentiles)
        table_data = df_task.round(3).reset_index()
        headers = ['Task', 'Count', 'Mean', 'Std', 'Min']
        headers += [f'{int(100*p)}%' for p in percentiles]
        headers += ['Max']
        print(tabulate(table_data, headers, tablefmt='fancy_grid'))
        tables[f'{task_name}_data'] = tabulate(table_data, headers, tablefmt='latex')
    return tables


def save_latex_tables(latex_tables: Dict[str, str], artifacts_dir_str: str) -> None:
    artifacts_dir = pathlib.Path(artifacts_dir_str)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for table_name, table_latex in latex_tables.items():
        with open(artifacts_dir / f'{table_name}_statistics.txt', 'w') as f:
            f.write(table_latex)


def _read_subdirectory_dataset_csvs(datasets_dir_str: str) -> pd.DataFrame:
    datasets_dir = get_root_dir() / datasets_dir_str
    if not datasets_dir.is_dir():
        raise FileNotFoundError(f'Directory: {datasets_dir.resolve()}')

    dfs = []
    for dataset_dir in datasets_dir.iterdir():
        if dataset_dir.is_dir():
            train_file = dataset_dir / 'train.csv'
            if not train_file.exists():
                train_file = dataset_dir / 'valid.csv'
            if not train_file.exists():
                continue

            with open(train_file) as f:
                train_data = pd.read_csv(f)

            with open(dataset_dir / 'test.csv') as f:
                test_data = pd.read_csv(f)

            train_data['split'] = 'train'
            test_data['split'] = 'test'

            data = pd.concat([train_data, test_data])
            data['task'] = dataset_dir.name
            dfs.append(data)
    return pd.concat(dfs)


if __name__ == '__main__':
    main()
