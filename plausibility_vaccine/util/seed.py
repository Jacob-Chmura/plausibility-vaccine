import logging
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    logging.info('Setting seed: %s', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
