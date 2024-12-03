import argparse
import pathlib

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description='Generate Selectional Association Data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--svo-probes-file',
    type=str,
    default='../data/verb_understanding_data/svo_probes.csv',
    help='Path to file containing raw svo probes data in CSV format',
)
parser.add_argument(
    '--subject-save-dir',
    type=str,
    default='../data/verb_understanding_data/selectional_association_subject',
    help='Path to save subject selectional association data',
)
parser.add_argument(
    '--object-save-dir',
    type=str,
    default='../data/verb_understanding_data/selectional_association_object',
    help='Path to save object selectional association data',
)
parser.add_argument(
    '--test-frac',
    type=float,
    default=0.2,
    help='The proportion of the dataset to include in the test split',
)
parser.add_argument(
    '--rng',
    type=int,
    default=1337,
    help='The random state to use when splitting data, for reproducibility',
)


def main() -> None:
    args = parser.parse_args()
    probe_data = pathlib.Path(args.svo_probes_file)
    df = pd.read_csv(probe_data, usecols=['pos_triplet'])

    # Process the pos_triplet column
    df['pos_triplet'] = df['pos_triplet'].apply(lambda x: x.split(','))
    df = pd.DataFrame(df['pos_triplet'].tolist(), columns=['subject', 'verb', 'object'])

    sa_subject = compute_selectional_association(df, col_name='subject')
    sa_object = compute_selectional_association(df, col_name='object')

    save_data(
        sa_subject,
        test_frac=args.test_frac,
        save_dir_str=args.subject_save_dir,
        rng=args.rng,
    )
    save_data(
        sa_object,
        test_frac=args.test_frac,
        save_dir_str=args.object_save_dir,
        rng=args.rng,
    )


def compute_selectional_association(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    data = pd.merge(
        df.groupby('verb')[col_name].value_counts(normalize=True).reset_index(),
        df[col_name].value_counts(normalize=True).reset_index(),
        how='left',
        on=col_name,
    ).rename({'proportion_x': 'p_given_v', 'proportion_y': 'p'}, axis=1)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        return sum(p * np.log(p / q))

    sr_data = (
        data.groupby('verb')
        .apply(lambda x: kl(x['p_given_v'], x['p']), include_groups=False)
        .reset_index()
        .rename({0: 'selectional_preference'}, axis=1)
    )

    data = pd.merge(data, sr_data, how='left', on='verb')
    data = data[data['selectional_preference'] > 0]
    data['label'] = data['p_given_v'] * np.log(data['p_given_v'] / data['p'])
    data['label'] /= data['selectional_preference']
    return data[['verb', col_name, 'label']]


def save_data(df: pd.DataFrame, test_frac: float, save_dir_str: str, rng: int) -> None:
    save_dir = pathlib.Path(save_dir_str)
    save_dir.mkdir(parents=True, exist_ok=True)

    df_test = df.sample(frac=test_frac, random_state=rng)
    df_train = df.drop(df_test.index)

    df_train.to_csv(save_dir / 'train.csv', index=False)
    df_test.to_csv(save_dir / 'test.csv', index=False)


if __name__ == '__main__':
    main()
