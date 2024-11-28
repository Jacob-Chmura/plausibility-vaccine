import logging
from typing import Tuple, Union

from adapters import AdapterConfig, AutoAdapterModel
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from plausibility_vaccine.util import seed_everything, setup_basic_logging

seed = 0
pretrained_model_name = 'albert-base-v2'


def load_pretrained_model(
    model_name: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    logging.info('Loading AutoTokenizer for: %s', model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logging.info('Loading AutoModel for: %s', model_name)
    model = AutoAdapterModel.from_pretrained(model_name)

    return model, tokenizer


def add_adapter_head(
    model: PreTrainedModel,
    adapter_name: str,
    adapter_config: Union[str, AdapterConfig],
) -> PreTrainedModel:
    model.add_adapter(adapter_name, config=adapter_config)
    return model


def add_plausibility_adapter_head(model: PreTrainedModel) -> PreTrainedModel:
    model.add_classification_head('plausibility', num_labels=2)
    return model


def main() -> None:
    setup_basic_logging()
    seed_everything(seed)

    model, tokenizer = load_pretrained_model(pretrained_model_name)
    model = add_plausibility_adapter_head(model)


if __name__ == '__main__':
    main()
