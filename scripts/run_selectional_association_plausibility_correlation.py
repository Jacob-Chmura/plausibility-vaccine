import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter

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

    get_merged_plausibility_sa_data(plausibility_df, sa_datasets_dir)
    # plot_correlation(sa_df, args.artifacts_dir)

    joint_sa_df = get_joint_merged_plausibility_sa_data(
        plausibility_df, sa_datasets_dir
    )
    plot_joint_correlation(joint_sa_df, args.artifacts_dir)


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


def get_joint_merged_plausibility_sa_data(
    plausibility_df: pd.DataFrame, sa_datasets_dir: pathlib.Path
) -> pd.DataFrame:
    dfs = []
    for task, plausibility_task in plausibility_df.groupby('task'):
        data_dir = sa_datasets_dir / task
        sa_df = _read_subdirectory_dataset_csvs(data_dir)

        plausibility_task['sv_association'] = plausibility_task.apply(
            lambda x: sa_df[
                (sa_df['subject'] == x['subject']) & (sa_df['verb'] == x['verb'])
            ]['label'].values[0],
            axis=1,
        )
        plausibility_task['vo_association'] = plausibility_task.apply(
            lambda x: sa_df[
                (sa_df['object'] == x['object']) & (sa_df['verb'] == x['verb'])
            ]['label'].values[0],
            axis=1,
        )
        plausibility_task = plausibility_task.rename({'label': 'plausibility'}, axis=1)
        plausibility_task = plausibility_task[
            ['task', 'sv_association', 'vo_association', 'plausibility']
        ]
        dfs.append(plausibility_task)
    df = pd.concat(dfs)
    return df


def plot_joint_correlation(df: pd.DataFrame, artifacts_dir_str: str) -> pd.DataFrame:
    artifacts_dir = pathlib.Path(artifacts_dir_str)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df['sv_group'] = (100 * df['sv_association'].round(1)).astype(int)
    df['vo_group'] = (100 * df['vo_association'].round(1)).astype(int)
    df = df[(df['sv_group'] >= 0) & (df['vo_group'] >= 0)]
    df2 = df.groupby(['sv_group', 'vo_group'])['plausibility'].mean()
    df2 = df2.reset_index()
    df2.groupby('sv_group')
    corr = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            res = df2[(df2.sv_group == 10 * i) & (df2.vo_group == 10 * j)][
                'plausibility'
            ].values
            corr[i][j] = res[0] if len(res) else 0.5

    corr = gaussian_filter(corr, sigma=1.7)
    a = sns.heatmap(
        corr,
        cmap='coolwarm_r',
        vmin=0.43,
        vmax=0.57,
        cbar_kws={'label': 'Plausibility Probability'},
    )

    a.set_xlabel('Subject-Verb Association')
    a.set_xticks([0, 5, 10])
    a.set_xticklabels([0, 0.5, 1])
    a.set_ylabel('Verb-Object Association')
    a.set_yticks([0, 5, 10])
    a.set_yticklabels([0, 0.5, 1])

    plt.savefig(
        artifacts_dir / 'joint_selectional_association_correlation.png',
        dpi=200,
        bbox_inches='tight',
    )
    plt.close()


def plot_correlation(df: pd.DataFrame, artifacts_dir_str: str) -> pd.DataFrame:
    artifacts_dir = pathlib.Path(artifacts_dir_str)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for entity, df_g in df.groupby('entity_type'):
        color_map = 'Greens_r' if entity == 'subject' else 'Blues_r'
        c = sns.color_palette(color_map, 13)[1]
        a = sns.regplot(
            data=df_g,
            x='association',
            y='plausibility',
            label=entity,
            scatter_kws={'s': 2, 'alpha': 0.3},
            color=c,
            line_kws=dict(color=c),
            lowess=True,
        )
        a.grid(alpha=0.3)
        a.set_ylabel('Plausibility')
        a.set_xlabel('Verb Association')
        a.spines[['right', 'top']].set_visible(False)
        a.plot([0], [0], c=c, label=entity[0].upper() + entity[1:])
        a.legend()

    ax = plt.gca()
    h, l = ax.get_legend_handles_labels()
    ax.legend(
        handles=[h[1], h[3]],
        labels=[l[1], l[3]],
        frameon=False,
        loc=(0.3, 0.1),
        ncol=2,
    )
    plt.savefig(
        artifacts_dir / 'selectional_association_correlation.png',
        dpi=200,
        bbox_inches='tight',
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
