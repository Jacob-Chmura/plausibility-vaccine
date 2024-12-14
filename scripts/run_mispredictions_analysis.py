import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import wordnet

from plausibility_vaccine.util.path import get_root_dir

parser = argparse.ArgumentParser(
    description='Misprediction analysis.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--data-dir',
    type=str,
    default='data/plausibility_prop_assoc_data/',
    help='Path to directory containing the concatanted misprediction plausibility data.',
)
parser.add_argument(
    '--figures-dir',
    type=str,
    default='figs/misprediction_analysis',
    help='Path to write misprediction figures.',
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
    '--association-datasets-dir',
    type=str,
    default='data/verb_understanding_data/',
    help='Path to root directory containing property data',
)
parser.add_argument(
    '--results-dir',
    type=str,
    default='results/albert_reduce_factor_64/',
    help='Path to to results directory for albert_64.',
)
parser.add_argument(
    '--output-dir',
    type=str,
    default='data/plausibility_prop_assoc_data/',
    help='Path to to output directory containing combined data',
)


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
                data = pd.read_csv(f)

            data['split'] = 'train'

            data['task'] = dataset_dir.name
            dfs.append(data)
    return pd.concat(dfs)


def get_merged_plausibility_property_data(
    plausibility_data: pd.DataFrame, property_data: pd.DataFrame
) -> pd.DataFrame:
    plausibility_data = plausibility_data[plausibility_data.split == 'train']
    plausibility_data = plausibility_data.rename({'label': 'plausibility'}, axis=1)
    property_data['Item'] = property_data['Item'].apply(lambda x: x.lower())

    dfs = []
    for property_name, df_property in property_data.groupby('task'):
        property = df_property[['Item', 'label']]
        for entity_name in ['subject', 'object']:
            entity = plausibility_data[
                [
                    entity_name,
                    'task',
                    'plausibility',
                    'baseline_predict',
                    'endline_predict',
                    'random_predict',
                    f'{entity_name}_sense_rank',
                ]
            ].rename({entity_name: 'Item'}, axis=1)
            entity = pd.merge(entity, property, how='left', on='Item').dropna()
            entity['property'] = property_name
            entity['entity'] = entity_name

            dfs.append(entity)

    df = pd.concat(dfs)
    df['property_label'] = df['label'].astype(int)
    return df


def get_merged_plausibility_sa_data(
    plausibility_df: pd.DataFrame, sa_datasets_dir: pathlib.Path
) -> pd.DataFrame:
    dfs = []
    for task, plausibility_task in plausibility_df.groupby('task'):
        for entity in ['subject', 'object']:
            entity_df = plausibility_task[
                [
                    entity,
                    'verb',
                    'label',
                    'baseline_predict',
                    'endline_predict',
                    'random_predict',
                    f'{entity}_sense',
                ]
            ]
            entity_df = entity_df.rename({'label': 'plausibility'}, axis=1)

            data_dir = sa_datasets_dir / task
            sa_df = _read_subdirectory_dataset_csvs(data_dir)

            sa_df = sa_df[[entity, 'verb', 'label']].dropna()
            sa_df = pd.merge(sa_df, entity_df, how='left', on=[entity, 'verb'])
            sa_df = sa_df.rename({'label': 'association', entity: 'Item'}, axis=1)
            sa_df['entity'] = entity
            sa_df['task'] = task
            dfs.append(sa_df)
    df = pd.concat(dfs)
    return df


def bucket_distribution_mispredictions(df: pd.DataFrame) -> pd.DataFrame:
    # List to store the results for each group
    dfs = []

    for (label, name), property_df in df.groupby(['property_label', 'property']):
        endline_mispredictions = (
            property_df['endline_predict'] != property_df['plausibility']
        ).sum()

        baseline_mispredictions = (
            property_df['baseline_predict'] != property_df['plausibility']
        ).sum()

        random_mispredictions = (
            property_df['random_predict'] != property_df['plausibility']
        ).sum()

        group_summary = pd.DataFrame(
            {
                'property': [name],
                'property_label': [label],
                'endline_mispredictions': [endline_mispredictions],
                'baseline_mispredictions': [baseline_mispredictions],
                'random_mispredictions': [random_mispredictions],
                'total_samples': [len(property_df)],
            }
        )

        dfs.append(group_summary)
    results = pd.concat(dfs, ignore_index=True)
    return results


def add_predictions_sense_rank(
    df: pd.DataFrame, baseline_results: pathlib.Path, endline_results: pathlib.Path
) -> pd.DataFrame:
    def _sense_rank(word: str, word_sense: str) -> int:
        synsets = [syn.name() for syn in wordnet.synsets(word, pos='n')]
        return synsets.index(word_sense) if word_sense in synsets else 0

    baseline = pd.read_csv(baseline_results)['predicted_label']
    endline = pd.read_csv(endline_results)['predicted_label']
    df['baseline_predict'] = baseline
    df['endline_predict'] = endline
    df['random_predict'] = np.random.randint(
        0, 2, df.shape[0]
    )  # Need to seed this somehow
    ranks = []
    for entity in ['subject', 'object']:
        for row in df[[entity, f'{entity}_sense']].iterrows():
            word = row[1][entity]
            sense = row[1][f'{entity}_sense']
            rank = _sense_rank(word, sense)
            ranks.append(rank)
        df[f'{entity}_sense_rank'] = ranks
        ranks = []
    return df


def hist_property_mispredictions(df: pd.DataFrame) -> None:
    for name, property_df in df.groupby('property'):
        x = property_df['property_label']
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(
            x,
            property_df['endline_mispredictions'],
            width,
            label='Endline Mispredictions',
        )
        ax.set_xlabel('Property Label')
        ax.set_ylabel('Misprediction Counts')
        ax.set_title(f'Endline Mispredictions for {name} across label strength')
        ax.set_xticks(x)
        ax.set_xticklabels(x)
        ax.grid(True, linestyle='--', color='gray', linewidth=0.7, which='both')
        ax.legend()

        plt.tight_layout()
        plt.show()


def hist_property_mispredictions_endline(df: pd.DataFrame) -> None:
    properties = df['property'].unique()
    labels = sorted(df['property_label'].unique())

    width = 0.1
    x = np.array(labels)

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, property_name in enumerate(properties):
        property_df = df[df['property'] == property_name]
        endline_counts = (
            property_df.set_index('property_label')
            .reindex(labels)['endline_mispredictions']
            .fillna(0)
        )
        ax.bar(x + i * width, endline_counts, width, label=property_name)

    ax.set_xlabel('Property Label')
    ax.set_ylabel('Misprediction Counts')
    ax.set_title('Endline Mispredictions for All Properties Across Label Strength')
    ax.set_xticks(x + (len(properties) * width) / 2 - width / 2)
    ax.set_xticklabels(labels)
    ax.grid(True, linestyle='--', color='gray', linewidth=0.7, which='both')
    ax.legend(title='Properties', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def mispredictions_on_sense(df: pd.DataFrame) -> None:
    mispredictions = df[df['plausibility'] != df['endline_predict']]
    mispredictions['rank'] = (
        mispredictions[['subject_sense_rank', 'object_sense_rank']]
        .bfill(axis=1)
        .iloc[:, 0]
    )
    mispredictions = mispredictions.dropna(subset=['rank'])
    rank_counts = mispredictions['rank'].astype(int)
    rank_counts = rank_counts[rank_counts != 0].value_counts().sort_index()
    rank_relative_freq = rank_counts / rank_counts.sum()

    plt.figure(figsize=(8, 6))
    plt.bar(rank_relative_freq.index, rank_relative_freq.values, width=0.7)
    plt.xlabel('Sense Rank')
    plt.ylabel('Relative Frequency of Mispredictions')
    plt.title('Frequency of mispredictions over sense ranks.')
    plt.xticks(rank_relative_freq.index)
    plt.grid(axis='y', linestyle='--', color='grey', linewidth=0.7)
    plt.tight_layout()
    plt.show()


def joint_corr_table(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))

    corr = df[
        [
            'association',
            'plausibility',
            'random_predict',
            'endline_predict',
        ]
    ].corr()

    annot = corr.round(2).astype(str)
    annot = annot.where(pd.notnull(corr), 'NA')

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=annot,
        fmt='',
        cmap='coolwarm',
        linewidths=0.5,
        mask=corr.isnull(),
        cbar_kws={'label': 'Correlation'},
        annot_kws={'size': 12, 'weight': 'bold'},
    )

    sns.heatmap(
        corr.isnull(),
        cmap=sns.light_palette('yellow', as_cmap=True),
        cbar=False,
        linewidths=0.5,
        annot=annot,
        fmt='',
        mask=~corr.isnull(),
    )

    plt.title('Correlation Heatmap')
    plt.show()


def main() -> None:
    args = parser.parse_args()
    association_dir = get_root_dir() / args.association_datasets_dir

    plausibility_df = _read_subdirectory_dataset_csvs(
        args.plausibility_datasets_dir
    )  # Already combined pep/20q

    property_df = _read_subdirectory_dataset_csvs(args.property_datasets_dir)
    baseline_file = (
        get_root_dir()
        / args.results_dir
        / '20q_plausibility_combined_finetune_adapter_base'
        / 'train_binary_predictions.csv'
    )

    endline_file = (
        get_root_dir()
        / args.results_dir
        / '20q_plausibility_combined_finetune_adapter_use_adapters'
        / 'train_binary_predictions.csv'
    )
    plausibility_df = add_predictions_sense_rank(
        plausibility_df, baseline_file, endline_file
    )
    merged_property_df = get_merged_plausibility_property_data(
        plausibility_df, property_df
    )
    merged_association_df = get_merged_plausibility_sa_data(
        plausibility_df, association_dir
    )
    mispredictions_on_sense(merged_property_df)
    joint_corr_table(merged_association_df)
    bins = bucket_distribution_mispredictions(merged_property_df)
    hist_property_mispredictions_endline(bins)
    hist_property_mispredictions(bins)


if __name__ == '__main__':
    main()
