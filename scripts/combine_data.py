import argparse
import pathlib

import pandas as pd


def combine_csv(
    file_1: pathlib.Path, file_2: pathlib.Path, output_file: pathlib.Path
) -> None:
    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    output_file.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure output directory exists
    combined_df.to_csv(output_file, index=False)


def main() -> None:
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Combine downstream datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--pep3k-datasets-dir',
        type=str,
        default='../data/plausibility_data/pep_3k/',
        help='Path to root directory containing plausibility data',
    )
    parser.add_argument(
        '--qa-datasets-dir',
        type=str,
        default='../data/plausibility_data/twentyquestions/',
        help='Path to root directory containing QA data',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/plausibility_data_combined/',
        help='Path to root directory for combined output data',
    )
    args = parser.parse_args()

    pep3k_dir = pathlib.Path(args.pep3k_datasets_dir)
    qa_dir = pathlib.Path(args.qa_datasets_dir)
    output_dir = pathlib.Path(args.output_dir)

    # File paths for valid data
    file_1_valid = pep3k_dir / 'valid.csv'
    file_2_valid = qa_dir / 'valid.csv'
    output_file_valid = output_dir / 'combined_valid.csv'
    combine_csv(file_1_valid, file_2_valid, output_file_valid)

    # File paths for test data
    file_1_test = pep3k_dir / 'test.csv'
    file_2_test = qa_dir / 'test.csv'
    output_file_test = output_dir / 'combined_test.csv'
    combine_csv(file_1_test, file_2_test, output_file_test)

    print(f'Datasets combined and saved to {output_dir}')


if __name__ == '__main__':
    main()
