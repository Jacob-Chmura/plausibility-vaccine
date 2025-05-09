import argparse

import pandas as pd

from plausibility_vaccine.util.path import get_root_dir

parser = argparse.ArgumentParser(
    description='Generate Prompt Data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--prompt-file',
    type=str,
    default='data/property_data/prompt_data.csv',
    help='Path to file containing raw prompt data in CSV format',
)
parser.add_argument(
    '--save-root-dir',
    type=str,
    default='data/property_data/',
    help='Path to root direcotry to save all property datasets',
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
    prompt_data = get_root_dir() / args.prompt_file
    df = pd.read_csv(prompt_data)

    df_test = df.sample(frac=args.test_frac, random_state=args.rng)
    df_train = df.drop(df_test.index)

    property_cols = [c for c in df.columns if c != 'Item']
    for property in property_cols:
        save_dir = get_root_dir() / args.save_root_dir / property
        save_dir.mkdir(parents=True, exist_ok=True)

        prop_train = df_train[['Item', property]].rename({property: 'label'}, axis=1)
        prop_test = df_test[['Item', property]].rename({property: 'label'}, axis=1)

        prop_train.to_csv(save_dir / 'train.csv', index=False)
        prop_test.to_csv(save_dir / 'test.csv', index=False)


if __name__ == '__main__':
    main()
