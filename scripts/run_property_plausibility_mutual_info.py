import argparse
import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

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
    '--artifacts-dir',
    type=str,
    default='artifacts',
    help='Path to artifact directory containing plots',
)


def main() -> None:
    args = parser.parse_args()
    df = get_merged_plausibility_property_data(
        args.plausibility_datasets_dir, args.property_datasets_dir
    )
    mi_df = compute_mutual_info_scores(df)
    plot_mi_df(mi_df, args.artifacts_dir)


def get_merged_plausibility_property_data(
    plausibility_datasets_dir: str, property_datasets_dir: str
) -> pd.DataFrame:
    plausibility_data = _read_subdirectory_dataset_csvs(plausibility_datasets_dir)
    property_data = _read_subdirectory_dataset_csvs(property_datasets_dir)

    plausibility_data = plausibility_data[plausibility_data.split == 'train']
    plausibility_data = plausibility_data.rename({'label': 'plausibility'}, axis=1)
    property_data['Item'] = property_data['Item'].apply(lambda x: x.lower())

    dfs = []
    for property_name, df_property in property_data.groupby('task'):
        property = df_property[['Item', 'label']]
        for entity_name in ['subject', 'object']:
            entity = plausibility_data[[entity_name, 'task', 'plausibility']].rename(
                {entity_name: 'Item'}, axis=1
            )
            entity = pd.merge(entity, property, how='left', on='Item').dropna()
            entity['property'] = property_name
            entity['entity'] = entity_name

            dfs.append(entity)

    df = pd.concat(dfs)
    df['property_label'] = df['label'].astype(int)
    return df[['task', 'plausibility', 'property_label', 'property', 'entity']]


def compute_mutual_info_scores(df: pd.DataFrame, n_samples: int = 50) -> pd.DataFrame:
    mi_data: Dict[str, List[float]] = {
        'task': [],
        'entity': [],
        'property': [],
        'mi': [],
    }
    groups = ['task', 'entity', 'property']
    for _ in range(n_samples):
        for (task, entity, property), df_g in df.groupby(groups):
            x, y = df_g[['property_label']], df_g['plausibility']
            mi = mutual_info_classif(x, y)[0]
            mi_data['task'].append(task)
            mi_data['entity'].append(entity)
            mi_data['property'].append(property)
            mi_data['mi'].append(mi)
    mi_df = pd.DataFrame(mi_data)
    mi_df = mi_df.groupby(groups)['mi'].mean().reset_index()
    mi_df2 = (
        mi_df.groupby('task')['mi'].max().reset_index().rename({'mi': 'max_mi'}, axis=1)
    )
    mi_df = pd.merge(mi_df, mi_df2, how='left', on=['task'])
    mi_df['norm_mi'] = mi_df['mi'] / mi_df['max_mi']
    return mi_df


def plot_mi_df(df: pd.DataFrame, artifacts_dir_str: str) -> None:
    artifacts_dir = pathlib.Path(artifacts_dir_str)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df['group'] = df.apply(
        lambda x: x.entity[0].upper() + x.entity[1:] + ' ' + x.property, axis=1
    )
    df2 = (
        df[['group', 'norm_mi', 'task']]
        .groupby('group')['norm_mi']
        .mean()
        .reset_index()
        .sort_values(by='norm_mi', ascending=False)
    )
    labels = df2.group.values

    palette_object = sns.color_palette('Blues_r', 13)
    palette_subject = sns.color_palette('Greens_r', 13)
    palette = []
    o_idx, s_idx = 0, 0
    for l in labels:
        if l.startswith('Subject'):
            palette.append(palette_subject[s_idx])
            s_idx += 1
        else:
            palette.append(palette_object[o_idx])
            o_idx += 1

    g = sns.catplot(
        data=df2,
        x='norm_mi',
        y='group',
        kind='bar',
        palette=palette,
        legend=False,
    )
    for _, a in enumerate(g.axes.flatten()):
        a.grid(alpha=0.3)
        a.set_ylabel('')
        a.set_xlabel('Normalized Mutual Information')
        a.set_xlim(0, 1)

    plt.legend(title='', loc='center right', labels=['Object', 'Subject'])
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legend_handles[0].set_color(palette_object[1])
    leg.legend_handles[1].set_color(palette_subject[1])

    plt.savefig(
        artifacts_dir / 'property_mutual_info.png', dpi=200, bbox_inches='tight'
    )
    plt.close()


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
