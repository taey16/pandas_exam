from typing import Tuple, List, Dict

import numpy as np
import pandas as pd

import torch

from dataset.dataset import (
    read_csv,
    group_by_newID_and_normalize_row,
    shuffle_by_key,
    train_val_split_by_key,
    print_data_info,
    build_dataloader,
)

from models.model import get_train_model
from utils import AverageMeter, set_random_seed
from sorry_op import sorry_op

import experiment


@torch.no_grad()
def validate(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: str = "cuda:0",
    amp: bool = True
) -> Tuple[float, float]:

    loss = 0.
    correct = 0
    accuracy = 0.
    sample_counter = 0.
    for idx, (inputs, targets) in enumerate(loader):
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = inputs.shape[0]
        seq_length = inputs.shape[1]

        with torch.autocast("cuda", enabled=amp):
            logits = model(inputs)
            # logits.shape: (B, seq_length, 2)
            loss += torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            ).sum()
            pred = torch.softmax(logits, dim=2)
            pred = torch.argmax(pred, dim=2)

            # onehot to label
            targets = torch.argmax(targets, dim=2)
            correct += (pred == targets).sum()
            sample_counter += batch_size * seq_length

    accuracy = float(correct / sample_counter)
    loss = float(loss / sample_counter)

    return accuracy, loss


def run_train(
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    lr: float = 0.00001,
    weight_decay: float = 1e-9,
    batch_size: int = 128,
    max_epochs: int = 100,
    attention_head: int = 2,
    attention_dim_base: int = 64,
    attention_depth = 2,
    emb_dropout = 0.5,
    device: str = "cuda:0",
    amp: bool = False,
):

    disp_freq = 50
    max_grad_norm = 5.0

    best_acc = 0.0
    best_loss = 10000.

    # Get dataloader
    train_loader = build_dataloader(
        train_data, batch_size=batch_size, mode="train"
    )
    val_loader = build_dataloader(
        val_data, batch_size=batch_size, mode="val"
    )
    test_loader = build_dataloader(
        test_data, batch_size=batch_size, mode="test"
    )

    # Check datasets
    assert train_loader.dataset.feature_dim == val_loader.dataset.feature_dim
    assert train_loader.dataset.feature_dim == test_loader.dataset.feature_dim
    assert train_loader.dataset.seq_length == val_loader.dataset.seq_length
    assert train_loader.dataset.seq_length == test_loader.dataset.seq_length

    # Attention dim should be devided by attention_head without residual.
    attention_dim = attention_dim_base * attention_head
    # Get input-dim and seq_length from dataset
    dim_in = train_loader.dataset.feature_dim
    max_seq_len = train_loader.dataset.seq_length

    # Get model, optimizer, amp-scaler, and lr_scheduler
    model, optimizer, scheduler, amp_scaler = get_train_model(
        dim_in=dim_in,
        dim_out=2, # binary-classification
        attention_dim=attention_dim,
        attention_depth=attention_depth,
        attention_head=attention_head,
        max_seq_len=max_seq_len,
        lr=lr,
        max_epochs=max_epochs,
        optimizer_name="adamw",
        amp=amp,
        device=device
    )

    global_iters = 0
    loss_avgmeter = AverageMeter()
    # Main loop
    for epoch in range(max_epochs):
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # inputs.shape: (B, seq_length, feature_dim)

            with torch.autocast("cuda", enabled=amp):
                # Forward
                logits = model(inputs)
                # logits.shape: (B, seq_length, 2)

                # Compute loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="mean"
                )

            # Backward
            if amp_scaler is not None:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
            else:
                loss.backward()

            # Clip grad-norm, if necessary
            if max_grad_norm > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
            else:
                grad_norm = -1.0

            # Update grad.
            if amp_scaler is not None:
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                optimizer.step()

            # Update iter.
            global_iters += 1

            # Console flushing
            if idx % disp_freq == 0:
                # Get prior prob. p(c==1) in a batch
                num_neg = int(targets[:,:,0].sum()) / 10
                num_pos = int(targets[:,:,1].sum()) / 10
                assert num_neg + num_pos == batch_size
                prior_pos = (num_pos / batch_size) * 100

                # Get info.
                loss = float(loss)
                loss_avgmeter.update(loss)
                grad_norm = float(grad_norm)
                amp_scale = float(amp_scaler.get_scale()) \
                    if amp_scaler is not None else 0
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"ep: {epoch} it: {global_iters} "\
                    f"loss: {loss:.2f}({loss_avgmeter.avg:.7f}), "\
                    f"g: {grad_norm:.4f}, s: {amp_scale}, lr: {lr:.6f} "\
                    f"#pos: {num_pos}, #neg: {num_neg}, random_guess: {prior_pos:.0f}",
                    flush=True
                )

        # Update per epoch
        scheduler.step()

        # Validating
        accuracy, loss = validate(val_loader, model, device=device, amp=amp)
        print(f"ep: {epoch}, ACC: {accuracy * 100:.4f}, LOSS: {loss:.2f}", flush=True) 

        if best_acc < accuracy:
            best_acc = accuracy
            best_loss = loss
            print(
                f"Best so far: "\
                f"BEST-ACC: {best_acc * 100:.4f}, "\
                f"BEST-LOSS: {best_loss:.2f} "\
                f"in ep {epoch}",
                flush=True
            )

    return best_acc, best_loss


def run_experiment(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
    threshold_corr: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    attention_head: int,
    attention_dim_base: int,
    attention_depth: int,
    emb_dropout: float,
    device: str = "cuda:0",
    amp: bool = False,
) -> None:

    # Get data info.
    data_info_dict, useless_column_name = print_data_info(
        train_data, threshold_corr=threshold_corr
    )
    # Add useless columns
    drop_column_name += useless_column_name

    # Convert a pandas table into a Dict grouped by newID 
    train_data = group_by_newID_and_normalize_row(
        train_data,
        data_info_dict=data_info_dict,
        drop_label_name=drop_column_name,
        mode="train",
        use_dump_file=False
    )
    test_data = group_by_newID_and_normalize_row(
        test_data,
        data_info_dict=data_info_dict,
        drop_label_name=drop_column_name,
        mode="test",
        use_dump_file=False
    )
    # Shuffle
    train_data = shuffle_by_key(train_data)
    # Train/Val split
    train_data, val_data = train_val_split_by_key(
        train_data, use_dump_file=False
    )

    # Training
    best_acc, best_loss = run_train(
        train_data, val_data, test_data,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        attention_head=attention_head,
        attention_dim_base=attention_dim_base,
        attention_depth=attention_depth,
        emb_dropout=emb_dropout,
        device=device,
        amp=amp
    )

    return best_acc, best_loss, drop_column_name


def main(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
    device: str = "cuda:0",
    amp: bool = False
) -> bool:

    # Conduct experiments (grid-search)
    experiment.exp_grid_search(
        train_data, test_data,
        drop_column_name,
        fn_run_experiment=run_experiment,
        device=device,
        amp=amp
    )
    """
    experiment.exp_grid_search_full(
        train_data, test_data,
        drop_column_name,
        fn_run_experiment=run_experiment,
        device=device,
        amp=amp
    )
    """

    
if __name__ == "__main__":
    set_random_seed(0)

    train_csv = "inputs/abusingDetectionTrainDataset.csv"
    test_csv = "inputs/abusingDetectionTestDataset.csv"

    train_data = read_csv(train_csv)
    test_data = read_csv(test_csv)
    # Values for these columns will be dropped in later.
    drop_column_name = ["newID", "logging_timestamp"]

    try:
        # Get started to train a model to detect suspicious users.
        # We designed this task as to solve a sequence classification problem.
        main(train_data, test_data, drop_column_name)
    except Exception as e:
        print(e)
        sorry_op(torch.cuda.current_device())
