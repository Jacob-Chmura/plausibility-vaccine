from typing import List

from adapters import AdapterArguments, setup_adapter_training
from transformers import PreTrainedModel

from plausibility_vaccine.util.args import AdapterArguments


def setup_adapters(
    model: PreTrainedModel,
    adapter_args: AdapterArguments,
    task_name: str,
    label_list: List[str],
) -> PreTrainedModel:
    model.add_classification_head(
        task_name,
        num_labels=len(label_list),
        id2label={i: v for i, v in enumerate(label_list)},
    )
    model.add_adapter(task_name, config=adapter_args.adapter_config)
    model.set_active_adapters(task_name)

    setup_adapter_training(model, adapter_args, task_name)

    if adapter_args.train_adapter:
        model.train_adapter(task_name)

    return model
