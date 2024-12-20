import argparse
import pathlib
from typing import Dict

import pandas as pd
from nltk.corpus import wordnet
from tabulate import tabulate

from plausibility_vaccine.util.path import get_root_dir

pd.options.display.max_rows = 100

parser = argparse.ArgumentParser(
    description='Run Word Sense Analysis of Plausibility Datasets',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--plausibility-datasets-dir',
    type=str,
    default='data/plausibility_data/',
    help='Path to root directory containing plausibility data',
)
parser.add_argument(
    '--artifacts-dir',
    type=str,
    default='artifacts/word_sense_statistics',
    help='Path to artifact directory containing plots',
)


def main() -> None:
    args = parser.parse_args()
    df = _read_subdirectory_dataset_csvs(args.plausibility_datasets_dir)
    latex_tables = compute_sense_rank_latex_tables(df)
    save_latex_tables(latex_tables, args.artifacts_dir)


def compute_sense_rank_latex_tables(df: pd.DataFrame, top_n: int = 3) -> Dict[str, str]:
    def _sense_rank(x: pd.Series, col: str) -> int:
        word, word_sense = x[col], x[f'{col}_sense']
        synsets = [syn.name() for syn in wordnet.synsets(word, pos='n')]
        return synsets.index(word_sense) if word_sense in synsets else 0

    tables = {}
    for col in ['subject', 'object']:
        df[f'{col}_synset_rank'] = df.apply(lambda x: _sense_rank(x, col), axis=1)
        groups = df.groupby('task')[f'{col}_synset_rank'].value_counts(normalize=True)
        groups = groups.reset_index()

        rank_groups = []
        for g, df_g in groups.groupby('task'):
            g_top_n = df_g[df_g[f'{col}_synset_rank'].isin(list(range(top_n)))]
            g_other = pd.DataFrame(
                {
                    'task': [g],
                    f'{col}_synset_rank': ['Other'],
                    'proportion': [1 - g_top_n['proportion'].sum()],
                }
            )
            g_all = pd.concat([g_top_n, g_other])
            rank_groups.append(g_all)

        rank_df = pd.concat(rank_groups)
        rank_df = rank_df.pivot(index='task', columns=f'{col}_synset_rank')
        table_data = rank_df['proportion'].round(2)

        headers = [col] + [f'Sense Rank {i+1}' for i in range(top_n)] + ['Other']
        print(tabulate(table_data, headers=headers, tablefmt='fancy_grid'))
        tables[col] = tabulate(table_data, headers=headers, tablefmt='latex')
    return tables


def save_latex_tables(latex_tables: Dict[str, str], artifacts_dir_str: str) -> None:
    artifacts_dir = pathlib.Path(artifacts_dir_str)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for table_name, table_latex in latex_tables.items():
        with open(artifacts_dir / f'{table_name}_word_sense_dist.txt', 'w') as f:
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
