from typing import Tuple, List, Dict

import os
import gc

import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import (
    read_csv,
    group_by_newID_and_normalize_row,
    shuffle_by_key,
    train_val_split_by_key,
    get_data_info,
    build_dataloader,
)

from models.model import get_train_model
from utils import AverageMeter, set_random_seed, report_summary
from sorry_op import sorry_op

import experiment



@torch.no_grad()
def testing(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    output_filename: str,
    device: str = "cuda:0",
    amp: bool = True
) -> None:

    # A procedure for evaluating on the test dataset
    # There is no ground truth

    model.eval()

    output_fp = open(output_filename, "w")
    output_fp.write("newID, res\n")
    for idx, (inputs, newID) in enumerate(loader):
        inputs  = inputs.to(device, non_blocking=True)
        newID = newID[:, 0]

        with torch.autocast("cuda", enabled=amp):
            logits = model(inputs)
            # Aggregating logits accross time dimension, and then performing softmax. 
            pred = torch.softmax(logits.sum(1), dim=1)
            pred = torch.argmax(pred, dim=1)

        for sample_idx, _pred in enumerate(pred):
            output_fp.write(f"{int(newID[sample_idx])}, {int(_pred)}\n")

    output_fp.close()
    print(f"TESING DONE IN {output_filename}", flush=True)


@torch.no_grad()
def validate(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: str = "cuda:0",
    amp: bool = True
) -> Tuple[float, float]:

    # A procedure for evaluating on the val dataset
    # Perform to compute accuracy and loss

    model.eval()

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

            # one-hot to label
            targets = torch.argmax(targets, dim=2)
            correct += (pred == targets).sum()
            sample_counter += batch_size * seq_length

    accuracy = float(correct / sample_counter)
    loss = float(loss / sample_counter)

    return accuracy, loss


def run_infer(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
    exp_id: str,
    threshold_corr: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    optimizer_name: str,
    attention_head: int,
    attention_dim_base: int,
    attention_depth: int,
    emb_dropout: float,
    rel_pos_bias: bool,
    use_abs_pos_emb: bool,
    scaled_sinu_pos_emb: bool,
    post_emb_norm: bool,
    device: str = "cuda:0",
    amp: bool = False,
) -> None:

    # TH0.25-LR1e-05-WD1e-06-BS16-EP200-HEAD4-BASE64-D2-DROP0.5-ABSPOSTrue-RELPOSFalse-SCPOSFalse-NORMPOSTrue-posembed
    #TH0.30-LR5e-05-WD1e-06-BS32-EP200-HEAD6-BASE64-D2-DROP0.5-bs1632-head68
    #model_path = os.path.join("test/models", exp_id, "model.pth")
    #model_path = os.path.join("renew/models", exp_id, "model.pth")
    model_path = os.path.join("reborn/models", exp_id, "model.pth")
    #model_path = os.path.join("backup/models", exp_id, "model.pth")
    if not os.path.exists(model_path):
        assert 0, f"check model_path: {model_path}"
    else:
        state_dict = torch.load(model_path, map_location="cpu")
        print(f"model is loaded from {model_path}")

    # Get data info.
    data_info_dict, useless_column_name, all_column_name = get_data_info(
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

    # Get dataloader
    #train_loader = build_dataloader(
    #    train_data, batch_size=batch_size, mode="train"
    #)
    val_loader = build_dataloader(
        val_data, batch_size=batch_size, mode="val"
    )
    test_loader = build_dataloader(
        test_data, batch_size=batch_size, mode="test"
    )

    # Check datasets
    #assert train_loader.dataset.feature_dim == val_loader.dataset.feature_dim
    #assert train_loader.dataset.feature_dim == test_loader.dataset.feature_dim
    #assert train_loader.dataset.seq_length == val_loader.dataset.seq_length
    #assert train_loader.dataset.seq_length == test_loader.dataset.seq_length

    import pdb; pdb.set_trace()
    # Num. of attention dim should be divided by num. of attention_head without residual.
    attention_dim = attention_dim_base * attention_head
    # Get input-dim and seq_length from dataset
    dim_in = val_loader.dataset.feature_dim
    max_seq_len = val_loader.dataset.seq_length
    # Get model, optimizer, amp-scaler, and lr_scheduler
    model, _, _, _ = get_train_model(
        dim_in=dim_in,
        dim_out=2, # binary-classification
        attention_dim=attention_dim,
        attention_depth=attention_depth,
        attention_head=attention_head,
        max_seq_len=max_seq_len,
        emb_dropout=emb_dropout,
        rel_pos_bias=rel_pos_bias,
        use_abs_pos_emb=use_abs_pos_emb,
        scaled_sinu_pos_emb=scaled_sinu_pos_emb,
        post_emb_norm=post_emb_norm,
        lr=lr,
        max_epochs=max_epochs,
        optimizer_name=optimizer_name, 
        amp=amp,
        device=device
    )
    model.load_state_dict(state_dict)
    model.to(device)

    # Validating
    accuracy, loss = validate(val_loader, model, device=device, amp=amp)

    """
    # Training
    testing(
        test_loader, model,
        output_filename=output_csv_filename,
        device=device, amp=amp
    )
    """


def main(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
    device: str = "cuda:0",
    amp: bool = False
) -> bool:

    import pdb; pdb.set_trace()
    # Conduct experiments (grid-search)
    experiment.infer(
        train_data, test_data,
        drop_column_name,
        run_infer,
        note="posembed",
        device=device,
        amp=amp
    )

    
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
        # We designed this task as adressing a sequence classification problem.
        main(train_data, test_data, drop_column_name)
    except Exception as e:
        print(e)
        sorry_op(torch.cuda.current_device())
