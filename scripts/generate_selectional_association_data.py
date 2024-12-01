import argparse
import pathlib
from collections import defaultdict
from typing import DefaultDict, Tuple

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

    verb_subject_counts: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    verb_object_counts: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    subject_counts: DefaultDict[str, int] = defaultdict(int)
    object_counts: DefaultDict[str, int] = defaultdict(int)
    verb_counts: DefaultDict[str, int] = defaultdict(int)

    # Process the pos_triplet column
    for triplet in df['pos_triplet']:
        if isinstance(triplet, str):
            subject, verb, obj = triplet.split(',')

            # Count occurrences
            verb_subject_counts[(verb, subject)] += 1
            verb_object_counts[(verb, obj)] += 1
            subject_counts[subject] += 1
            object_counts[obj] += 1
            verb_counts[verb] += 1

    # Selectional Association
    sa_subject = compute_selectional_association(
        verb_counts, verb_subject_counts, subject_counts, x_name='subject'
    )

    sa_object = compute_selectional_association(
        verb_counts, verb_object_counts, object_counts, x_name='object'
    )
    print(len(sa_subject))
    print(args.subject_save_file)

    sa_subject.to_csv(args.subject_save_file, index=False)
    sa_object.to_csv(args.object_save_file, index=False)


def compute_selectional_association(
    verb_counts: DefaultDict[str, int],
    verb_x_counts: DefaultDict[Tuple[str, str], int],
    x_counts: DefaultDict[str, int],
    x_name: str,
) -> pd.DataFrame:
    # Compute probabilities
    p_x_given_v = {
        (verb, subject): count / verb_counts[verb]
        for (verb, subject), count in verb_x_counts.items()
    }
    total_x = sum(x_counts.values())
    p_x = {x: count / total_x for x, count in x_counts.items()}

    # Compute S_R(v)
    s_r: DefaultDict[str, float] = defaultdict(float)
    for verb in verb_counts.keys():
        s_r[verb] = sum(
            p_x_given_v[(verb, x)] * np.log(p_x_given_v[(verb, x)] / p_x.get(x, 1e-10))
            for (v, x) in p_x_given_v.keys()
            if v == verb
        )

    # Compute selectional association
    s_a: DefaultDict[Tuple[str, str], float] = defaultdict(float)
    for (verb, x), p_s_v in p_x_given_v.items():
        if s_r[verb] > 0:
            s_a[(verb, x)] = p_s_v * np.log(p_s_v / p_x.get(x, 1e-10)) / s_r[verb]

    s_a_df: pd.DataFrame = pd.DataFrame.from_dict(
        s_a, orient='index', columns=['association']
    ).reset_index()
    s_a_df[['verb', x_name]] = pd.DataFrame(s_a_df['index'].tolist())
    return s_a_df[['verb', x_name, 'association']]


if __name__ == '__main__':
    main()
