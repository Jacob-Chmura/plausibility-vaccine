import argparse
import pathlib

import pandas as pd

pd.options.display.max_rows = 100

parser = argparse.ArgumentParser(
    description='Run analysis of datasets',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--plausibility-datasets-dir',
    type=str,
    default='../data/plausibility_data/',
    help='Path to root directory containing plausibility data',
)
parser.add_argument(
    '--property-datasets-dir',
    type=str,
    default='../data/property_data/',
    help='Path to root directory containing property data',
)
parser.add_argument(
    '--verb-datasets-dir',
    type=str,
    default='../data/verb_understanding_data/',
    help='Path to root directory containing verb data',
)


def main() -> None:
    args = parser.parse_args()
    analyze_plausibility_datasets(args.plausibility_datasets_dir)
    analyze_property_datasets(args.property_datasets_dir)
    analyze_verb_datasets(args.verb_datasets_dir)


def analyze_plausibility_datasets(datasets_dir: str) -> None:
    df = _read_subdirectory_dataset_csvs(datasets_dir)
    print(df.groupby(['task', 'split'])['label'].value_counts(normalize=True))


def analyze_property_datasets(datasets_dir: str) -> None:
    df = _read_subdirectory_dataset_csvs(datasets_dir)
    for _, df_task in df.groupby('task'):
        print(
            df_task.groupby(['task'])['label']
            .value_counts(normalize=True)
            .reset_index()
            .sort_values('label')
        )


def analyze_verb_datasets(datasets_dir: str) -> None:
    df = _read_subdirectory_dataset_csvs(datasets_dir)
    print(df.head())


def _read_subdirectory_dataset_csvs(datasets_dir_str: str) -> pd.DataFrame:
    datasets_dir = pathlib.Path(datasets_dir_str)
    if not datasets_dir.is_dir():
        raise FileNotFoundError(f'Directory: {datasets_dir.resolve()}')

    dfs = []
    for dataset_dir in datasets_dir.iterdir():
        if dataset_dir.is_dir():
            train_file = dataset_dir / 'train.csv'
            if not train_file.exists():
                train_file = dataset_dir / 'valid.csv'

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
