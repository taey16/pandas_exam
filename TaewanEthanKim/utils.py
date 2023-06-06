
import gc
import random

import numpy as np

import torch


def set_random_seed(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # In case of single gpu
    torch.cuda.manual_seed(seed)
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


def report_summary(
    writer: torch.utils.tensorboard.SummaryWriter,
    epoch: int,
    iters: int,
    phase: str,
    total_iters: int = None,
    loss: float = None,
    accuracy: float = None,
    lr: float = None,
    grad_norm: float = None,
    **kwargs
) -> None:

    if loss is not None:
        writer.add_scalar(f"{phase}/loss", loss, iters)
    if accuracy is not None:
        writer.add_scalar(f"{phase}/accuracy", accuracy, iters)
    if lr is not None:
        writer.add_scalar("meta/lr", lr, iters)
    if grad_norm is not None:
        writer.add_scalar("meta/grad_norm", grad_norm, iters)


def clear_object(obj: object) -> None:
    if obj is not None:
        del obj
        gc.collect()
        torch.cuda.empty_cache()
        obj = None
        return obj
    else:
        return obj
