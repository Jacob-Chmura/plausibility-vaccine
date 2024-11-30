import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

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
class DataArguments:
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


@dataclass
class FinetuningArgument:
    data_args: DataArguments = field(
        metadata={'help': 'Data arguments for the fine-tuning configuration'},
    )
    adapter_args: AdapterArguments = field(
        metadata={'help': 'Adapter arguments for the fine-tuning configuration'},
    )


@dataclass
class FinetuningArguments:
    finetuning_args: Dict[str, FinetuningArgument] = field(
        metadata={'help': 'List of fine-tuning arguments'},
    )

    def __post_init__(self) -> None:
        for finetune_name, finetune_args in self.finetuning_args.items():
            self.finetuning_args[finetune_name] = FinetuningArgument(
                data_args=DataArguments(
                    **finetune_args['FinetuningArgument']['data_args']  # type: ignore
                ),
                adapter_args=AdapterArguments(
                    **finetune_args['FinetuningArgument']['adapter_args']  # type: ignore
                ),
            )


def parse_args(
    config_yaml: Union[str, pathlib.Path],
) -> Tuple[
    MetaArguments,
    ModelArguments,
    TrainingArguments,
    FinetuningArguments,
]:
    config_dict = yaml.safe_load(pathlib.Path(config_yaml).read_text())
    config_dict = (
        config_dict['MetaArguments']
        | config_dict['ModelArguments']
        | config_dict['TrainingArguments']
        | config_dict['FinetuningArguments']
    )
    parser = HfArgumentParser(
        (
            MetaArguments,
            ModelArguments,
            TrainingArguments,
            FinetuningArguments,
        )
    )
    return parser.parse_dict(config_dict, allow_extra_keys=True)
