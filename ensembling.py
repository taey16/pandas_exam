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
from trainer import testing, validate
from utils import AverageMeter, set_random_seed, report_summary
from sorry_op import sorry_op

import experiment


def run_infer(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
    exp_id: str,
    rnd_seed: int,
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

    model_path = os.path.join("test/models", exp_id, "model.pth")
    #model_path = os.path.join("renew/models", exp_id, "model.pth")
    #model_path = os.path.join("reborn/models", exp_id, "model.pth")
    #model_path = os.path.join("backup/models", exp_id, "model.pth")
    #model_path = os.path.join("final_posemb/models", exp_id, "model.pth")
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
    train_data = shuffle_by_key(train_data, rnd_seed=rnd_seed)
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
    rnd_seed: int,
    device: str = "cuda:0",
    amp: bool = False
) -> bool:

    import pdb; pdb.set_trace()
    # Conduct experiments (grid-search)
    experiment.infer(
        train_data, test_data,
        drop_column_name,
        run_infer,
        rnd_seed=rnd_seed,
        note="posembed-reprod-final",
        device=device,
        amp=amp
    )

    
if __name__ == "__main__":
    rnd_seed = 0
    set_random_seed(rnd_seed)

    train_csv = "inputs/abusingDetectionTrainDataset_lastpadding.csv"
    test_csv = "inputs/abusingDetectionTestDataset_lastpadding.csv"

    train_data = read_csv(train_csv)
    test_data = read_csv(test_csv)
    # Values for these columns will be dropped in later.
    drop_column_name = ["newID", "logging_timestamp"]

    try:
        # Get started to train a model to detect suspicious users.
        # We designed this task as adressing a sequence classification problem.
        main(train_data, test_data, drop_column_name, rnd_seed)
    except Exception as e:
        print(e)
        sorry_op(torch.cuda.current_device())
