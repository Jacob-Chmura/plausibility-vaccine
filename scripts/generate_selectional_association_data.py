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

    total_subjects = sum(subject_counts.values())
    total_objects = sum(object_counts.values())

    # Compute probabilities
    p_s_given_v = {
        (verb, subject): count / verb_counts[verb]
        for (verb, subject), count in verb_subject_counts.items()
    }
    p_s = {subject: count / total_subjects for subject, count in subject_counts.items()}

    p_o_given_v = {
        (verb, obj): count / verb_counts[verb]
        for (verb, obj), count in verb_object_counts.items()
    }
    p_o = {obj: count / total_objects for obj, count in object_counts.items()}

    ##Selectional Association

    selectional_association_subject = defaultdict(float)
    selectional_association_object = defaultdict(float)

    # Compute S_R(v) for subjects
    s_r_subject = {}
    for verb in verb_counts.keys():
        s_r_subject[verb] = sum(
            p_s_given_v[(verb, subject)]
            * np.log(p_s_given_v[(verb, subject)] / p_s.get(subject, 1e-10))
            for (v, subject) in p_s_given_v.keys()
            if v == verb
        )

    # Compute S_R(v) for objects
    s_r_object = {}
    for verb in verb_counts.keys():
        s_r_object[verb] = sum(
            p_o_given_v[(verb, obj)]
            * np.log(p_o_given_v[(verb, obj)] / p_o.get(obj, 1e-10))
            for (v, obj) in p_o_given_v.keys()
            if v == verb
        )

    # Compute selectional association for subjects
    for (verb, subject), p_s_v in p_s_given_v.items():
        if s_r_subject[verb] > 0:
            selectional_association_subject[(verb, subject)] = (
                p_s_v * np.log(p_s_v / p_s.get(subject, 1e-10)) / s_r_subject[verb]
            )

    # Compute selectional association for objects
    for (verb, obj), p_o_v in p_o_given_v.items():
        if s_r_object[verb] > 0:
            selectional_association_object[(verb, obj)] = (
                p_o_v * np.log(p_o_v / p_o.get(obj, 1e-10)) / s_r_object[verb]
            )

    selectional_association_df_subject = pd.DataFrame.from_dict(
        selectional_association_subject,
        orient='index',
        columns=['Selectional_Association'],
    ).reset_index()
    selectional_association_df_subject.columns = [
        'Verb-Subject',
        'Selectional_Association',
    ]

    selectional_association_df_object = pd.DataFrame.from_dict(
        selectional_association_object,
        orient='index',
        columns=['Selectional_Association'],
    ).reset_index()
    selectional_association_df_object.columns = [
        'Verb-Object',
        'Selectional_Association',
    ]

    selectional_association_df_subject.to_csv(args.subject_save_file, index=False)
    selectional_association_df_object.to_csv(args.object_save_file, index=False)


if __name__ == '__main__':
    main()
