import logging
from typing import Dict, List, Optional, Tuple

from datasets import DatasetDict, load_dataset
from transformers import BatchEncoding, PreTrainedTokenizer

from plausibility_vaccine.util.args import DataArguments


def preprocess_function(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    label_list: List[str],
) -> BatchEncoding:
    # Tokenize the texts
    args = examples['subject'], examples['verb'], examples['object']
    result = tokenizer(*args, padding='max_length', max_length=8, truncation=True)

    # Map labels to IDs
    label_to_id = {v: i for i, v in enumerate(label_list)}
    if label_to_id is not None and 'label' in examples:
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

    if data_args.is_regression:
        label_list = None
        logging.info(f'{data_args.task_name}.is_regression = True')
    else:
        label_list = raw_datasets['train'].unique('label')
        label_list.sort()  # Let's sort it for determinism
        logging.info(f'{data_args.task_name} label list: {label_list}')

    return raw_datasets, label_list
