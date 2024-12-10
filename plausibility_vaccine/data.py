import logging
from typing import Dict, List, Optional, Tuple

from datasets import DatasetDict, concatenate_datasets, load_dataset
from transformers import BatchEncoding, PreTrainedTokenizer

from plausibility_vaccine.util.args import DataArguments


def preprocess_function(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    label_list: Optional[List[str]],
) -> BatchEncoding:
    y_col = 'label'
    # TODO: Make this configurable instead of hard coding these heuristics
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
    logging.info(f'Loading training dataset with files: {data_args.train_file}')
    train_dataset = load_dataset('csv', data_files=data_args.train_file)
    train_dataset = concatenate_datasets(train_dataset.values())

    logging.info(f'Loading testing dataset with files: {data_args.test_file}')
    test_dataset = load_dataset('csv', data_files=data_args.test_file)
    test_dataset = concatenate_datasets(test_dataset.values())

    raw_datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})
    logging.info('Loaded datasets: %s', raw_datasets)

    if data_args.is_regression:
        label_list = None
        logging.info(f'{data_args.task_name}.is_regression = True')
    else:
        label_list = raw_datasets['train'].unique('label')
        label_list.sort()  # Let's sort it for determinism
        logging.info(f'{data_args.task_name} label list: {label_list}')

    return raw_datasets, label_list
