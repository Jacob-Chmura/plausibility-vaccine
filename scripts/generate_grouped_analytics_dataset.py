import argparse
import pathlib

import pandas as pd

from plausibility_vaccine.util.path import get_root_dir

parser = argparse.ArgumentParser(
    description='Run mutual information analysis of property and plausibility',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--plausibility-datasets-dir',
    type=str,
    default='data/plausibility_data/',
    help='Path to root directory containing plausibility data',
)
parser.add_argument(
    '--property-datasets-dir',
    type=str,
    default='data/property_data/',
    help='Path to root directory containing property data',
)
parser.add_argument(
    '--selectional-association-datasets-dir',
    type=str,
    default='data/verb_understanding_data/',
    help='Path to root directory containing property data',
)
parser.add_argument(
    '--results-dir',
    type=str,
    default='results/albert_reduce_factor_64/',
    help='Path to to results directory for albert_64.',
)
parser.add_argument(
    '--output-dir',
    type=str,
    default='data/plausibility_prop_assoc_data/',
    help='Path to to output directory containing combined data',
)


def _read_subdirectory_dataset_csvs(datasets_dir_str: str) -> pd.DataFrame:
    datasets_dir = get_root_dir() / datasets_dir_str
    if not datasets_dir.is_dir():
        raise FileNotFoundError(f'Directory: {datasets_dir.resolve()}')

    dfs = []
    for dataset_dir in datasets_dir.iterdir():
        if dataset_dir.is_dir():
            train_file = dataset_dir / 'train.csv'
            if not train_file.exists():
                train_file = dataset_dir / 'valid.csv'

            with open(train_file) as f:
                data = pd.read_csv(f)

            data['split'] = 'train'

            data['task'] = dataset_dir.name
            dfs.append(data)
    return pd.concat(dfs)


def get_merged_plausibility_property_data(
    plausibility_data: pd.DataFrame, property_data: pd.DataFrame
) -> pd.DataFrame:
    plausibility_data = plausibility_data[plausibility_data.split == 'train']
    plausibility_data = plausibility_data.rename({'label': 'plausibility'}, axis=1)
    property_data['Item'] = property_data['Item'].apply(lambda x: x.lower())

    dfs = []
    for property_name, df_property in property_data.groupby('task'):
        property = df_property[['Item', 'label']]
        for entity_name in ['subject', 'object']:
            entity = plausibility_data[
                [
                    entity_name,
                    'task',
                    'plausibility',
                    'baseline_predict',
                    'endline_predict',
                ]
            ].rename({entity_name: 'Item'}, axis=1)
            entity = pd.merge(entity, property, how='left', on='Item').dropna()
            entity['property'] = property_name
            entity['entity'] = entity_name

            dfs.append(entity)

    df = pd.concat(dfs)
    df['property_label'] = df['label'].astype(int)
    return df


def get_merged_plausibility_sa_data(
    plausibility_df: pd.DataFrame, sa_datasets_dir: pathlib.Path
) -> pd.DataFrame:
    dfs = []
    for task, plausibility_task in plausibility_df.groupby('task'):
        for entity in ['subject', 'object']:
            entity_df = plausibility_task[
                [entity, 'verb', 'label', 'baseline_predict', 'endline_predict']
            ]
            entity_df = entity_df.rename({'label': 'plausibility'}, axis=1)

            data_dir = sa_datasets_dir / task
            sa_df = _read_subdirectory_dataset_csvs(data_dir)

            sa_df = sa_df[[entity, 'verb', 'label']].dropna()
            sa_df = pd.merge(sa_df, entity_df, how='left', on=[entity, 'verb'])
            sa_df = sa_df.rename({'label': 'association', entity: 'Item'}, axis=1)
            sa_df['entity'] = entity
            sa_df['task'] = task
            dfs.append(sa_df)
    df = pd.concat(dfs)
    return df


def main() -> None:
    args = parser.parse_args()
    output_dir_prop = get_root_dir() / args.output_dir / 'concate_properties.csv'
    output_dir_assoc = get_root_dir() / args.output_dir / 'concate_association.csv'
    sa_datasets_dir = get_root_dir() / args.selectional_association_datasets_dir

    plausibility_df = _read_subdirectory_dataset_csvs(
        args.plausibility_datasets_dir
    )  # Already combined pep/20q

    # Append the predictions
    # Baseline:
    baseline_file = (
        get_root_dir()
        / args.results_dir
        / '20q_plausibility_combined_finetune_adapter_base'
        / 'train_binary_predictions.csv'
    )
    baseline = pd.read_csv(baseline_file)['predicted_label']
    print(f'Shape of baseline: {baseline.shape}')
    sum_base = baseline.sum()
    print(f'sum of baseline: {sum_base}')

    # Endline
    endline_file = (
        get_root_dir()
        / args.results_dir
        / '20q_plausibility_combined_finetune_adapter_use_adapters'
        / 'train_binary_predictions.csv'
    )
    endline = pd.read_csv(endline_file)['predicted_label']
    print(f'Shape of endline: {endline.shape}')
    sum_end = endline.sum()
    print(f'sum of endline: {sum_end}')

    print(f'Shape of plausibility_df: {plausibility_df.shape}')

    plausibility_df['baseline_predict'] = baseline
    plausibility_df['endline_predict'] = endline
    print(plausibility_df.head())
    print(f'Shape of plausibility_df: {plausibility_df.shape}')

    property_df = _read_subdirectory_dataset_csvs(args.property_datasets_dir)
    print('Head of property_df:')
    print(property_df.head())
    df_merged_properties = get_merged_plausibility_property_data(
        plausibility_df, property_df
    )
    print(f'Shape of merged properties: {df_merged_properties.shape}')
    print(f'The duplicates: {df_merged_properties.duplicated().sum()}')
    baseline_sum = df_merged_properties['baseline_predict'].sum()
    print(f'Baseline sum merge: {baseline_sum}')
    endline_sum = df_merged_properties['endline_predict'].sum()
    print(f'Endline sum merge: {endline_sum}')
    df_merged_properties.to_csv(output_dir_prop)

    df_merged_assocation = get_merged_plausibility_sa_data(
        plausibility_df, sa_datasets_dir
    )
    df_merged_assocation.to_csv(output_dir_assoc)


if __name__ == '__main__':
    main()
