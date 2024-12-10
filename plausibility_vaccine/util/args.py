import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import yaml
from adapters import AdapterArguments
from transformers import HfArgumentParser, TrainingArguments

from plausibility_vaccine.util.path import get_root_dir


@dataclass
class MetaArguments:
    log_file_path: Optional[str] = field(
        metadata={'help': 'Path to the log file to use.'},
    )
    global_seed: int = field(
        default=1337,
        metadata={'help': 'Random seed to use for reproducibiility.'},
    )

    def __post_init__(self) -> None:
        if self.log_file_path is not None:
            self.log_file_path = str(get_root_dir() / self.log_file_path)


@dataclass
class DataArguments:
    task_name: str = field(
        metadata={'help': 'The name of the task to train on'},
    )
    is_regression: bool = field(
        metadata={'help': 'Is the task a regression or classification problem'},
    )
    train_file: Union[str, List[str]] = field(
        metadata={'help': 'A csv or list of csv files containing the training data.'},
    )
    test_file: Union[str, List[str]] = field(
        metadata={'help': 'A csv or list of csv files containing the test data.'},
    )
    num_test_cv: int = field(
        metadata={'help': 'Number of test splits to do for uncertainty estimates.'},
        default=1,
    )

    def __post_init__(self) -> None:
        root_dir = get_root_dir()

        def resolve_paths(files: Union[str, List[str]]) -> Union[str, List[str]]:
            if isinstance(files, str):
                return str(root_dir / files)
            return [str(root_dir / file) for file in files]

        self.train_file = resolve_paths(self.train_file)
        self.test_file = resolve_paths(self.test_file)


@dataclass
class ModelArguments:
    pretrained_model_name: str = field(
        metadata={'help': 'Pretrained model identifier from huggingface.co/models'},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'Directory to store pretrained models from huggingface.co'},
    )

    def __post_init__(self) -> None:
        if self.cache_dir is not None:
            self.cache_dir = str(get_root_dir() / self.cache_dir)


@dataclass
class FinetuningArgument:
    data_args: DataArguments = field(
        metadata={'help': 'Data arguments for the fine-tuning configuration'},
    )
    adapter_args: Optional[AdapterArguments] = field(
        metadata={'help': 'Adapter arguments for the fine-tuning configuration'},
    )
    use_adapter_for_task: bool = field(
        default=True,
        metadata={'help': 'Use an adapter for finetuning, or adds a dense nn layer'},
    )
    fusion: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'List of Adapters to fuse for fine-tuning configuration'},
    )


@dataclass
class FinetuningArguments:
    pretraining_tasks: Dict[str, FinetuningArgument] = field(
        metadata={'help': 'List of fine-tuning tasks to pre-train on'},
    )
    downstream_tasks: Dict[str, FinetuningArgument] = field(
        metadata={'help': 'List of fine-tuning tasks to downstream train on'},
    )

    def __post_init__(self) -> None:
        def _remap_nested_args(
            tasks: Dict[str, FinetuningArgument],
        ) -> Dict[str, FinetuningArgument]:
            for task_name, task_args in tasks.items():
                adapter_args = None
                if task_args['adapter_args'] is not None:  # type: ignore
                    adapter_args = AdapterArguments(**task_args['adapter_args'])  # type: ignore

                tasks[task_name] = FinetuningArgument(
                    data_args=DataArguments(**task_args['data_args']),  # type: ignore
                    adapter_args=adapter_args,
                    use_adapter_for_task=task_args.get('use_adapter_for_task', True),  # type: ignore
                    fusion=task_args.get('fusion'),  # type: ignore
                )
            return tasks

        self.pretraining_tasks = _remap_nested_args(self.pretraining_tasks)
        self.downstream_tasks = _remap_nested_args(self.downstream_tasks)


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
