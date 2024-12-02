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
) -> PreTrainedModel:
    if label_list is None:
        num_labels, id2label = 1, None
    else:
        num_labels, id2label = len(label_list), {i: v for i, v in enumerate(label_list)}

    logging.info('Adding classification head adapter for task: %s', task_name)
    model.add_classification_head(task_name, num_labels=num_labels, id2label=id2label)

    logging.info('Adding adapter for task: %s', task_name)
    setup_adapter_training(model, adapter_args, task_name)
    return model
