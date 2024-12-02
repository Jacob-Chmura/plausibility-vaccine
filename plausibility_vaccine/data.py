import logging
from typing import Dict, List, Tuple

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


def get_data(data_args: DataArguments) -> Tuple[DatasetDict, List[str]]:
    data_files = {
        'train': data_args.train_file,
        'test': data_args.test_file,
    }
    for key in data_files.keys():
        logging.info(f'load a local file for {key}: {data_files[key]}')

    raw_datasets: DatasetDict = load_dataset('csv', data_files=data_files)

    label_list = raw_datasets['train'].unique('label')
    label_list.sort()  # Let's sort it for determinism
    return raw_datasets, label_list
