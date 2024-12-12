import argparse

import numpy as np
import pandas as pd

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
    df['predictions'] = np.random.randint(0, 1, df.shape[0])
    return df


def main() -> None:
    args = parser.parse_args()
    data_dir = get_root_dir() / args.data_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Results directory: {data_dir.resolve()}')
    property_data_path = data_dir / 'concate_properties.csv'
    df_property = pd.read_csv(property_data_path)
    df_property = random_predictions(df_property)
    print(df_property.head())


if __name__ == '__main__':
    main()
