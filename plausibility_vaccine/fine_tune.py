import logging
from typing import List, Optional

from adapters import AdapterArguments, setup_adapter_training
from transformers import PreTrainedModel

from plausibility_vaccine.util.args import AdapterArguments


def setup_adapters(
    model: PreTrainedModel,
    adapter_args: AdapterArguments,
    task_name: str,
    label_list: Optional[List[str]],
    fusion_list: Optional[List[str]],
) -> PreTrainedModel:
    if fusion_list is None:
        return _setup_adapter_pretraining(model, adapter_args, task_name, label_list)
    else:
        return _setup_adapter_fusion(model, task_name, fusion_list)


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
    logging.info('Model Active Adapters: %s', model.active_adapters)
    return model


def _setup_adapter_fusion(
    model: PreTrainedModel, task_name: str, fusion_list: List[str]
) -> PreTrainedModel:
    for task in fusion_list:
        # TODO: Need to push config through
        logging.info('Loading pre-trained adapter for task: %s', task)
        model.load_adapter(task)

    logging.info('Adding fusion adapter for task: %s', task_name)
    model.add_adapter_fusion(fusion_list, 'dynamic')
    model.train_adapter_fusion(fusion_list)
    logging.info('Model Active Adapters: %s', model.active_adapters)
    return model
