import logging
from typing import Tuple

from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from plausibility_vaccine.util import seed_everything, setup_basic_logging

seed = 0
pretrained_model_name = 'albert-base-v2'


def load_pretrained_model(
    model_name: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    logging.info('Loading AutoTokenizer for: %s', model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logging.info('Loading AutoModel for: %s', model_name)
    model = AutoModel.from_pretrained(model_name)

    return model, tokenizer


def main() -> None:
    setup_basic_logging()
    seed_everything(seed)

    model, tokenizer = load_pretrained_model(pretrained_model_name)

    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    _ = model(**encoded_input)


if __name__ == '__main__':
    main()
