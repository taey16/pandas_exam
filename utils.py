import random
import numpy as np

import torch

def set_random_seed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # In case of single gpu
    #torch.cuda.manual_seed(seed)
    # Multi-gpu case
    torch.cuda.manual_seed_all(seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(
        self,
        val: torch.tensor,
        n: int = 1
    ) -> None:
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count

    def get_val(self) -> float:
        return self.val

    def get_avg(self) -> float:
        return self.avg
