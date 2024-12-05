import logging
from typing import List, Optional, Tuple

from adapters import AutoAdapterModel
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from plausibility_vaccine.util.args import ModelArguments


def load_pretrained_model(
    model_args: ModelArguments,
    label_list: Optional[List[str]],
    use_adapter_for_task: bool,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    logging.info(f'Loading pre-trained model: {model_args}, label_list: {label_list}')

    config = AutoConfig.from_pretrained(
        model_args.pretrained_model_name,
        num_labels=len(label_list) if label_list is not None else 1,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.pretrained_model_name,
        cache_dir=model_args.cache_dir,
    )

    if use_adapter_for_task:
        model = AutoAdapterModel.from_pretrained(
            model_args.pretrained_model_name,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.pretrained_model_name,
            config=config,
            cache_dir=model_args.cache_dir,
        )
        model = _freeze_pretrained_model_gradients(model)

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    if label_list is not None:
        label_to_id = {v: i for i, v in enumerate(label_list)}
        model.config.label2id = label_to_id
        model.config.id2label = {
            id: label for label, id in model.config.label2id.items()
        }

    logging.debug('Loaded pre-trained model: %s', model)
    logging.debug('Loaded pre-trained tokenizer: %s', tokenizer)
    return model, tokenizer


def _freeze_pretrained_model_gradients(model: PreTrainedModel) -> PreTrainedModel:
    logging.info('Freezing pretrained model weights')
    for name, param in model.named_parameters():
        if not name.startswith('classifier'):
            param.requires_grad = False
            logging.debug('Froze parameters: %s', name)
    return model
