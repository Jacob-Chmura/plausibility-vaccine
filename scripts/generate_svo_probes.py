import argparse
import pathlib

import pandas as pd

from plausibility_vaccine.util.path import get_root_dir

parser = argparse.ArgumentParser(
    description='Generate svo triples from plausiblity training data.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--pep-file',
    type=str,
    default='data/plausibility_data/pep_3k/',
    help='Path to file containing plausibility data pep.',
)
parser.add_argument(
    '--questions-file',
    type=str,
    default='data/plausibility_data/twentyquestions/',
    help='Path to file containing plausibility data 20q.',
)

parser.add_argument(
    '--output-file',
    type=str,
    default='data/verb_understanding_data/',
    help='Path to outpute file with all verb undertandting data.',
)


def preprocess_data_triplet(
    input_path: pathlib.Path, output_path: pathlib.Path
) -> None:
    df = pd.read_csv(input_path)
    df['pos_triplet'] = df.apply(
        lambda row: f"{row['subject']}, {row['verb']}, {row['object']}", axis=1
    )

    result_df = df[['pos_triplet']]

    result_df.to_csv(output_path, index=False)


def main() -> None:
    args = parser.parse_args()
    pep_3k = get_root_dir() / args.pep_file / 'valid.csv'
    pep_output = get_root_dir() / args.output_file / 'pep_triplets.csv'
    preprocess_data_triplet(pep_3k, pep_output)
    question_20 = get_root_dir() / args.questions_file / 'valid.csv'
    questions_output = get_root_dir() / args.output_file / '20q_triplets.csv'
    preprocess_data_triplet(question_20, questions_output)


if __name__ == '__main__':
    main()
