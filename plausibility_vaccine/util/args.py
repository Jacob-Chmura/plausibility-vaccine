import pathlib
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import yaml
from adapters import AdapterArguments
from transformers import HfArgumentParser, TrainingArguments


@dataclass
class DataTrainingArguments:
    task_name: str = field(
        metadata={'help': 'The name of the task to train on'},
    )
    train_file: str = field(
        metadata={'help': 'A csv file containing the training data.'},
    )
    test_file: str = field(
        metadata={'help': 'A csv file containing the test data.'},
    )


@dataclass
class ModelArguments:
    pretrained_model_name: str = field(
        metadata={'help': 'Pretrained model identifier from huggingface.co/models'},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Where do you want to store the pretrained models downloaded from huggingface.co'
        },
    )


def parse_args(
    config_yaml: Union[str, pathlib.Path],
) -> Tuple[ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments]:
    config_dict = yaml.safe_load(pathlib.Path(config_yaml).read_text())
    config_dict = (
        config_dict['ModelArguments']
        | config_dict['DataTrainingArguments']
        | config_dict['TrainingArguments']
        | config_dict['AdapterArguments']
    )
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments)
    )
    return parser.parse_dict(config_dict, allow_extra_keys=True)
