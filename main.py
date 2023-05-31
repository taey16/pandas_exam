from typing import Tuple

import numpy as np

#import pandas as pd
from sklearn.utils import shuffle

import torch

from dataset.dataset import (
    read_csv,
    train_val_split,
    print_dtype_min_max,
    build_dataloader,
    normalize_data,
    check_data
)

from model import get_train_model


def run_train(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    device: str = "cuda:0",
    amp: bool = False,
):

    max_grad_norm = 2.0
    max_epochs = 1000
    lr = 0.01
    batch_size = 1024

    train_data = normalize_data(train_data, method="minmax")
    val_data = normalize_data(val_data, method="minmax")
    test_data = normalize_data(test_data, method="minmax")

    check_data(train_data)
    check_data(val_data)
    check_data(test_data)

    train_loader = build_dataloader(train_data, batch_size=batch_size, mode="train")
    val_loader = build_dataloader(val_data, batch_size=batch_size, mode="val")
    test_loader = build_dataloader(test_data, batch_size=batch_size, mode="test")

    model, optimizer, scheduler, amp_scaler = get_train_model(
        lr=lr,
        max_epochs=max_epochs,
        amp=amp,
        device=device
    )

    global_iters = 0
    for epoch in range(max_epochs):
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            inputs = inputs[..., None]
            # inputs.shape: (B, seq_length, feature_dim)

            with torch.autocast("cuda", enabled=amp):
                logits = model(inputs)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="mean"
                )

            if amp_scaler is not None:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
            else:
                loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if amp_scaler is not None:
                amp_scaler.step(optimizer)
                amp_scaler.update()

            global_iters += 1

            if idx % 1 == 0:
                loss = float(loss)
                grad_norm = float(grad_norm)
                amp_scale = float(amp_scaler.get_scale()) if amp_scaler is not None else 0
                lr = optimizer.param_groups[0]["lr"]
                print(f"ep: {epoch} it: {global_iters} loss: {loss:.2f}, g: {grad_norm}, s: {amp_scale}, lr: {lr}")

        scheduler.step()
        

if __name__ == "__main__":
    train_csv = "inputs/abusingDetectionTrainDataset.csv"
    test_csv = "inputs/abusingDetectionTestDataset.csv"


    train_data = read_csv(train_csv)
    test_data = read_csv(test_csv)

    train_data = shuffle(train_data, random_state=0)
    train_data, val_data = train_val_split(train_data)

    print_dtype_min_max(train_data)

    train_data = train_data.drop(columns=["logging_timestamp", "newID"])
    val_data = val_data.drop(columns=["logging_timestamp", "newID"])
    test_data = test_data.drop(columns=["logging_timestamp", "newID"])

    train_data = train_data.to_numpy()
    val_data = val_data.to_numpy()
    test_data = test_data.to_numpy()

    num_pos = (train_data[:,-1] == 1).sum()
    num_neg = (train_data[:,-1] == 0).sum()

    print(f"(train_data[:,-1] == 1).sum(): {(train_data[:,-1] == 1).sum()}")
    print(f"(train_data[:,-1] == 0).sum(): {(train_data[:,-1] == 0).sum()}")

    run_train(train_data, val_data, test_data)
