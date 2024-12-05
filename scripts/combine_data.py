import pandas as pd


def combine_csv(file_1: str, file_2: str, output_file: str) -> None:
    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df.to_csv(output_file, index=False)


def main() -> None:
    file_1_valid = '../data/plausibility_data/pep_3k/valid.csv'
    file_2_valid = '../data/plausibility_data/twentyquestions/valid.csv'
    output_file_valid = '../data/plausibility_data_combined/combined_valid.csv'
    combine_csv(file_1_valid, file_2_valid, output_file_valid)

    file_1_test = '../data/plausibility_data/pep_3k/valid.csv'
    file_2_test = '../data/plausibility_data/twentyquestions/valid.csv'
    output_file_test = '../data/plausibility_data_combined/combined_test.csv'
    combine_csv(file_1_test, file_2_test, output_file_test)


if __name__ == '__main__':
    main()
