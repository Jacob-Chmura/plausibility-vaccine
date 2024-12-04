import argparse
import json
import pathlib

import pandas as pd

parser = argparse.ArgumentParser(
    description='Aggregate performance results',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--results-dir',
    type=str,
    default='../results',
    help='Path to root directory containing results',
)


def main() -> None:
    args = parser.parse_args()
    results_dir = pathlib.Path(args.results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f'Results directory: {results_dir.resolve()}')

    dfs = []
    for result_dir in results_dir.iterdir():
        if result_dir.is_dir():
            with open(result_dir / 'eval_results.json') as f:
                result_data = json.load(f)

            result_df = pd.DataFrame.from_dict(result_data, orient='index').T
            result_df['task'] = result_dir.name
            dfs.append(result_df)

    dfs = pd.concat(dfs)
    print(dfs)


if __name__ == '__main__':
    main()
