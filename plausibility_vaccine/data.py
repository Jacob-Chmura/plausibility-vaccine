import logging
from typing import Dict, List, Optional, Tuple

from datasets import DatasetDict, load_dataset
from transformers import BatchEncoding, PreTrainedTokenizer

from plausibility_vaccine.util.args import DataArguments


def preprocess_function(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    label_list: Optional[List[str]],
) -> BatchEncoding:
    # TODO: Make this configurable instead of hard coding these heuristics
    y_col = 'label'
    if 'subject_sense' in examples:
        x_cols = ['subject', 'verb', 'object']
    else:
        x_cols = [col for col in examples if col != y_col]

    # Tokenize the texts
    args = [examples[x_col] for x_col in x_cols]
    result = tokenizer(*args, padding='max_length', max_length=8, truncation=True)

    # Map labels to IDs
    if label_list is not None:
        label_to_id = {v: i for i, v in enumerate(label_list)}
        result['label'] = [label_to_id[l] for l in examples['label']]
    return result


def get_data(data_args: DataArguments) -> Tuple[DatasetDict, Optional[List[str]]]:
    data_files = {'train': data_args.train_file}
    if data_args.test_file is None:
        # TODO: Probably want to do our own train/test split with the single file
        logging.warning('Test file for %s is None', data_args.task_name)
    else:
        data_files['test'] = data_args.test_file

    for key in data_files.keys():
        logging.info(f'Loading a local file for {key}: {data_files[key]}')

    raw_datasets: DatasetDict = load_dataset('csv', data_files=data_files)
    logging.info('Loaded datasets: %s', raw_datasets)

    # TODO: Rename the column in the actual raw data
    raw_datasets = raw_datasets.rename_column('association', 'label')
    logging.warning('Renamed column "association" to "label"')

    if data_args.is_regression:
        label_list = None
        logging.info(f'{data_args.task_name}.is_regression = True')
    else:
        label_list = raw_datasets['train'].unique('label')
        label_list.sort()  # Let's sort it for determinism
        logging.info(f'{data_args.task_name} label list: {label_list}')

    return raw_datasets, label_list
