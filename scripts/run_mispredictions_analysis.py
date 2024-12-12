import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

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
    help='Path to writed misprediction figures.',
)


def random_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df['rand_predictions'] = np.random.randint(0, 1, df.shape[0])
    return df


def main() -> None:
    args = parser.parse_args()
    data_dir = get_root_dir() / args.data_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Results directory: {data_dir.resolve()}')
    assocation_data_path = data_dir / 'concate_association.csv'
    df_assoc = pd.read_csv(assocation_data_path)
    df_assoc = random_predictions(df_assoc)
    print(df_assoc.head())

    df_assoc['is_misprediction'] = (
        df_assoc['rand_predictions'] != df_assoc['plausibility']
    )
    mispredictions = df_assoc[df_assoc['is_misprediction']]
    correct_predictions = df_assoc[~df_assoc['is_misprediction']]

    misprediction_associations = mispredictions['association']
    correct_prediction_associations = correct_predictions['association']

    t_stat, p_value = ttest_ind(
        misprediction_associations, correct_prediction_associations
    )
    print(f't_state:: {t_stat}, p_value :: {p_value}')

    plt.figure(figsize=(10, 6))
    plt.hist(misprediction_associations, alpha=0.6, label='Mispredictions', bins=10)
    plt.hist(
        correct_prediction_associations, alpha=0.6, label='Correct predictions', bins=10
    )
    plt.xlabel('Association')
    plt.ylabel('Frequency')
    plt.title('Distribution of Associations for Mispredictions vs Correct Predictions')
    plt.legend()
    plt.grid()
    plt.show()

    sns.heatmap(
        df_assoc[['association', 'plausibility', 'rand_predictions']].corr(),
        annot=True,
        cmap='coolwarm',
    )
    plt.title('Correlation Heatmap')
    plt.show()

    plt.figure(figsize=(10, 6))

    corr = df_assoc[['association', 'plausibility', 'rand_predictions']].corr()

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

    plt.scatter(
        mispredictions.index,
        mispredictions['association'],
        color='red',
        label='Mispredictions',
        s=100,
    )

    plt.scatter(
        correct_predictions.index,
        correct_predictions['association'],
        color='green',
        label='Correct Predictions',
        s=100,
    )

    plt.xlabel('Index')
    plt.ylabel('Association')
    plt.title(
        'Scatter Plot of Association Values for Mispredictions vs Correct Predictions'
    )
    plt.legend()
    plt.grid()
    plt.show()

    sns.pairplot(df_assoc)
    plt.show()


if __name__ == '__main__':
    main()
