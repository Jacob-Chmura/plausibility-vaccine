import argparse
import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split


def combine_csv(
    file_1: pathlib.Path, file_2: pathlib.Path, output_file: pathlib.Path
) -> pd.DataFrame:
    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    return combined_df


def split_and_save(
    combined_df: pd.DataFrame,
    output_dir: pathlib.Path,
    valid_filename: str = 'new_valid.csv',
    test_filename: str = 'new_test.csv',
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    train_df, test_df = train_test_split(
        combined_df, test_size=test_size, random_state=random_state
    )
    valid_file = output_dir / valid_filename
    test_file = output_dir / test_filename
    train_df.to_csv(valid_file, index=False)
    test_df.to_csv(test_file, index=False)
    print(f'New validation dataset saved to {valid_file}')
    print(f'New test dataset saved to {test_file}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Combine and split datasets',
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
        default='../data/plausibility_data_combined_80_20/',
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
    combined_valid = combine_csv(file_1_valid, file_2_valid, output_file_valid)

    # File paths for test data
    file_1_test = pep3k_dir / 'test.csv'
    file_2_test = qa_dir / 'test.csv'
    output_file_test = output_dir / 'combined_test.csv'
    combined_test = combine_csv(file_1_test, file_2_test, output_file_test)

    # Combine valid and test datasets split into new valid and test
    combined_all = pd.concat([combined_valid, combined_test], ignore_index=True)
    split_and_save(combined_all, output_dir)

    print(f'Datasets combined, split, and saved to {output_dir}')


if __name__ == '__main__':
    main()
