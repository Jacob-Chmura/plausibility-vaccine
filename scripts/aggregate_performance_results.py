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

    plausibility_dfs, property_adapters_dfs, verb_adapters_dfs = [], [], []
    for result_dir in results_dir.iterdir():
        if result_dir.is_dir():
            with open(result_dir / 'eval_results.json') as f:
                result_data = json.load(f)

            result_df = pd.DataFrame.from_dict(result_data, orient='index').T
            result_df['task'] = result_dir.name
            if 'plausibility' in result_dir.name:
                plausibility_dfs.append(result_df)
            elif 'verb' in result_dir.name:
                verb_adapters_dfs.append(result_df)
            else:
                property_adapters_dfs.append(result_df)

    plausibility_dfs = pd.concat(plausibility_dfs).sort_values('task')
    property_adapters_dfs = pd.concat(property_adapters_dfs).sort_values('task')
    verb_adapters_dfs = pd.concat(verb_adapters_dfs).sort_values('task')

    print(plausibility_dfs)
    print(property_adapters_dfs)
    print(verb_adapters_dfs)


if __name__ == '__main__':
    main()
