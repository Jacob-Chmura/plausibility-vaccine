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
    '--subject-save-file',
    type=str,
    default='../data/verb_understanding_data/selectional_association_subject.csv',
    help='Path to save subject selectional association data in CSV format',
)
parser.add_argument(
    '--object-save-file',
    type=str,
    default='../data/verb_understanding_data/selectional_association_object.csv',
    help='Path to save object selectional association data in CSV format',
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

    sa_subject.to_csv(args.subject_save_file, index=False)
    sa_object.to_csv(args.object_save_file, index=False)


def compute_selectional_association(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    data = pd.merge(
        df.groupby('verb')[col_name].value_counts(normalize=True).reset_index(),
        df[col_name].value_counts(normalize=True).reset_index(),
        how='left',
        on=col_name,
    ).rename({'proportion_x': 'p_given_v', 'proportion_y': 'p'}, axis=1)

    def _sr(p_given_v: np.ndarray, p: np.ndarray) -> float:
        return sum(p_given_v * np.log(p_given_v / p))

    sr_data = (
        data.groupby('verb')
        .apply(lambda x: _sr(x['p_given_v'], x['p']), include_groups=False)
        .reset_index()
        .rename({0: 'selectional_preference'}, axis=1)
    )

    data = pd.merge(data, sr_data, how='left', on='verb')
    data = data[data['selectional_preference'] > 0]

    data['association'] = data['p_given_v'] * np.log(data['p_given_v'] / data['p'])
    data['association'] /= data['selectional_preference']
    return data[['verb', col_name, 'association']]


if __name__ == '__main__':
    main()
