import logging
from typing import List, Optional

import transformers


def setup_basic_logging(
    log_file_path: Optional[str] = None,
    log_file_logging_level: int = logging.DEBUG,
    stream_logging_level: int = logging.INFO,
) -> None:
    handlers: List[logging.Handler] = []

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_logging_level)
    handlers.append(stream_handler)

    if log_file_path is not None:
        file_handler = logging.FileHandler(filename=log_file_path)
        file_handler.setLevel(log_file_logging_level)
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] %(levelname)s [%(processName)s %(threadName)s %(name)s.%(funcName)s:%(lineno)d] %(message)s',
        handlers=handlers,
    )

    _configure_transformers_logging(handlers)


def _configure_transformers_logging(handlers: List[logging.Handler]) -> None:
    transformers.utils.logging.disable_default_handler()
    for handler in handlers:
        transformers.utils.logging.add_handler(handler)
