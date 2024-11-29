import pathlib
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

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
    overwrite_cache: bool = field(
        default=False,
        metadata={'help': 'Overwrite the cached preprocessed datasets or not.'},
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
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments)
    )
    return parser.parse_yaml_file(config_yaml)
