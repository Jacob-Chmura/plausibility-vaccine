import pathlib
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import yaml
from adapters import AdapterArguments
from transformers import HfArgumentParser, TrainingArguments


@dataclass
class MetaArguments:
    log_file_path: Optional[str] = field(
        metadata={'help': 'Path to the log file to use.'},
    )
    global_seed: int = field(
        default=1337,
        metadata={'help': 'Random seed to use for reproducibiility.'},
    )


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
        metadata={'help': 'Directory to store pretrained models from huggingface.co'},
    )


def parse_args(
    config_yaml: Union[str, pathlib.Path],
) -> Tuple[
    MetaArguments,
    ModelArguments,
    DataTrainingArguments,
    TrainingArguments,
    AdapterArguments,
]:
    config_dict = yaml.safe_load(pathlib.Path(config_yaml).read_text())
    config_dict = (
        config_dict['MetaArguments']
        | config_dict['ModelArguments']
        | config_dict['DataTrainingArguments']
        | config_dict['TrainingArguments']
        | config_dict['AdapterArguments']
    )
    parser = HfArgumentParser(
        (
            MetaArguments,
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
            AdapterArguments,
        )
    )
    return parser.parse_dict(config_dict, allow_extra_keys=True)
