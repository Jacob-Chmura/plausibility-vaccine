import logging
import pathlib
from typing import List, Optional

import adapters
from adapters import AdapterArguments, setup_adapter_training
from adapters.composition import Fuse
from transformers import PreTrainedModel

from plausibility_vaccine.util.args import AdapterArguments


def setup_adapters(
    model: PreTrainedModel,
    adapter_args: AdapterArguments,
    task_name: str,
    label_list: Optional[List[str]],
    fusion_list: Optional[List[str]],
    use_adapter_for_task: bool,
    output_dir: str,
) -> PreTrainedModel:
    if not use_adapter_for_task and fusion_list is not None:
        logging.info(
            'Using a non-adapters-hub native pre trained model, and trying to use adapter fusion. '
            'Wrapping pre-trained model with adapters.init() in order to make this work'
        )
        adapters.init(model)

    if fusion_list is not None:
        model = _setup_adapter_fusion(
            model, task_name, label_list, fusion_list, use_adapter_for_task, output_dir
        )
    elif use_adapter_for_task:
        model = _setup_adapter_pretraining(model, adapter_args, task_name, label_list)

    logging.debug('Full Model:\n%s', model)
    return model


def _setup_adapter_pretraining(
    model: PreTrainedModel,
    adapter_args: AdapterArguments,
    task_name: str,
    label_list: Optional[List[str]],
) -> PreTrainedModel:
    if label_list is None:
        num_labels, id2label = 1, None
    else:
        num_labels, id2label = len(label_list), {i: v for i, v in enumerate(label_list)}
    logging.info('Adding classification head adapter for task: %s', task_name)
    model.add_classification_head(task_name, num_labels=num_labels, id2label=id2label)

    logging.info('Adding adapter for task: %s', task_name)
    adapter_args.train_adapter = True
    setup_adapter_training(model, adapter_args, task_name)
    logging.info('Model Adapter Summary:\n%s', model.adapter_summary())
    return model


def _setup_adapter_fusion(
    model: PreTrainedModel,
    task_name: str,
    label_list: Optional[List[str]],
    fusion_list: List[str],
    use_adapter_for_task: bool,
    output_dir: str,
) -> PreTrainedModel:
    for task in fusion_list:
        adapter_weight_path = _get_adapter_weight_path(output_dir, task)
        logging.info('Loading pre-trained adapter: %s', adapter_weight_path)
        model.load_adapter(str(adapter_weight_path), with_head=False)

    # Add Classification Head
    if use_adapter_for_task:
        logging.info('Adding classification head adapter for task: %s', task_name)
        if label_list is None:
            num_labels, id2label = 1, None
        else:
            num_labels, id2label = (
                len(label_list),
                {i: v for i, v in enumerate(label_list)},
            )
        model.add_adapter(task_name)
        model.add_classification_head(
            task_name, num_labels=num_labels, id2label=id2label
        )
        model.train_adapter(task_name)

    logging.info('Adding fusion adapter for task: %s', task_name)
    model.add_adapter_fusion(fusion_list, 'dynamic')
    model.train_adapter_fusion(Fuse(*fusion_list))
    logging.info('Model Adapter Summary:\n%s', model.adapter_summary())
    return model


def _get_adapter_weight_path(output_dir: str, adapter_name: str) -> pathlib.Path:
    adapter_weight_path = pathlib.Path(output_dir) / adapter_name
    adapter_weight_path.mkdir(parents=True, exist_ok=True)
    return adapter_weight_path
