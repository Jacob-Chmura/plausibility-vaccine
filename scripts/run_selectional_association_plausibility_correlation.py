import argparse
import pathlib

import pandas as pd

from plausibility_vaccine.util.path import get_root_dir

pd.options.display.max_rows = 100

parser = argparse.ArgumentParser(
    description='Run correlation analysis of selectional association and plausibility',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--plausibility-datasets-dir',
    type=str,
    default='data/plausibility_data/',
    help='Path to root directory containing plausibility data',
)
parser.add_argument(
    '--selectional-association-datasets-dir',
    type=str,
    default='data/verb_understanding_data/',
    help='Path to root directory containing property data',
)
parser.add_argument(
    '--artifacts-dir',
    type=str,
    default='artifacts',
    help='Path to artifact directory containing plots',
)


def main() -> None:
    args = parser.parse_args()

    sa_datasets_dir = pathlib.Path(args.selectional_association_datasets_dir)
    if not sa_datasets_dir.is_dir():
        raise FileNotFoundError(
            f'Selectional Association directory: {sa_datasets_dir.resolve()}'
        )

    plausibility_df = _read_subdirectory_dataset_csvs(args.plausibility_datasets_dir)
    plausibility_df = plausibility_df[plausibility_df.split == 'train']

    sa_df = get_merged_plausibility_sa_data(plausibility_df, sa_datasets_dir)
    print(sa_df)


def get_merged_plausibility_sa_data(
    plausibility_df: pd.DataFrame, sa_datasets_dir: pathlib.Path
) -> pd.DataFrame:
    dfs = []
    for task, plausibility_task in plausibility_df.groupby('task'):
        for entity in ['subject', 'object']:
            entity_df = plausibility_task[[entity, 'verb', 'label']]
            entity_df = entity_df.rename({'label': 'plausibility'}, axis=1)

            data_dir = sa_datasets_dir / task
            sa_df = _read_subdirectory_dataset_csvs(data_dir)

            sa_df = sa_df[[entity, 'verb', 'label']].dropna()
            sa_df = pd.merge(sa_df, entity_df, how='left', on=[entity, 'verb'])
            sa_df = sa_df.rename({'label': 'association', entity: 'entity'}, axis=1)
            sa_df['entity_type'] = entity
            sa_df['task'] = task
            dfs.append(sa_df)
    df = pd.concat(dfs)
    return df


def _read_subdirectory_dataset_csvs(datasets_dir_str: str) -> pd.DataFrame:
    datasets_dir = get_root_dir() / datasets_dir_str
    if not datasets_dir.is_dir():
        raise FileNotFoundError(f'Directory: {datasets_dir.resolve()}')

    dfs = []
    for dataset_dir in datasets_dir.iterdir():
        if dataset_dir.is_dir():
            train_file = dataset_dir / 'train.csv'
            test_file = dataset_dir / 'test.csv'
            if not train_file.exists():
                train_file = dataset_dir / 'valid.csv'

            with open(train_file) as f:
                train_data = pd.read_csv(f)
                train_data['split'] = 'train'

            if test_file.exists():
                with open(dataset_dir / 'test.csv') as f:
                    test_data = pd.read_csv(f)
                test_data['split'] = 'test'
                data = pd.concat([train_data, test_data])
            else:
                data = train_data

            data['task'] = dataset_dir.name
            dfs.append(data)
    return pd.concat(dfs)


if __name__ == '__main__':
    main()
