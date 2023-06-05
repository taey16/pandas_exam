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
def testing_ensemble(
    loader: torch.utils.data.DataLoader,
    models: List[torch.nn.Module],
    output_filename: str,
    device: str = "cuda:0",
    amp: bool = True
) -> None:

    # A procedure for evaluating on the test dataset
    # There is no ground truth

    num_ensemble = len(models)

    for model in models:
        model.eval()

    output_fp = open(output_filename, "w")
    output_fp.write("newID, res\n")
    for idx, (inputs, newID) in enumerate(loader):
        inputs  = inputs.to(device, non_blocking=True)
        newID = newID[:, 0]

        with torch.autocast("cuda", enabled=amp):
            logits = None
            for model in models:
                _logits = model(inputs)
                if logits is None: logits = _logits
                else: logits += _logits
            logits /= num_ensemble
            # Aggregating logits accross time dimension, and then performing softmax. 
            pred = torch.softmax(logits.sum(1), dim=1)
            pred = torch.argmax(pred, dim=1)

        for sample_idx, _pred in enumerate(pred):
            output_fp.write(f"{int(newID[sample_idx])}, {int(_pred)}\n")

    output_fp.close()
    print(f"TESING DONE IN {output_filename}", flush=True)


@torch.no_grad()
def validate_ensemble(
    loader: torch.utils.data.DataLoader,
    models: List[torch.nn.Module],
    device: str = "cuda:0",
    amp: bool = True
) -> Tuple[float, float]:

    # A procedure for evaluating on the val dataset
    # Perform to compute accuracy and loss

    num_ensemble = len(models)

    for model in models:
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
            logits = None
            for model in models:
                _logits = model(inputs)
                if logits is None: logits = _logits
                else: logits += _logits
            logits /= num_ensemble
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


def main(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
    rnd_seed: int,
    device: str = "cuda:0",
    amp: bool = False
) -> bool:


    drop_column_name = ["newID", "logging_timestamp"]
    threshold_corr = 0.35
    rnd_seed = 0
    batch_size = [16, 16]

    # Get data info.
    data_info_dict, useless_column_name, all_column_name = get_data_info(
        train_data, threshold_corr=threshold_corr
    )
    # Add useless columns
    drop_column_name += useless_column_name
    # Get survival columns for visualization
    set_all_column = set(all_column_name)
    set_useless_column = set(useless_column_name)
    print(f"Survived columns (th: {threshold_corr}): {set_all_column - set_useless_column}")

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

    val_loader = build_dataloader(
        val_data, batch_size=batch_size[0], mode="val"
    )
    test_loader = build_dataloader(
        test_data, batch_size=batch_size[0], mode="test"
    )

    # TH0.35-LR1e-05-WD1e-06-BS16-EP200-HEAD4-BASE64-D2-DROP0.5-ABSPOSTrue-RELPOSFalse-SCPOSTrue-NORMPOSFalse-posembed-reprod-final-Mreload
    # TH0.35-LR1e-05-WD1e-06-BS16-EP200-HEAD4-BASE64-D2-DROP0.5-ABSPOSTrue-RELPOSFalse-SCPOSTrue-NORMPOSTrue-posembed-reprod-final-Mreload

    note = ["posembed-reprod-final-Mreload", "posembed-reprod-final-Mreload"]
    lr = [0.00001, 0.00001]
    attention_head = [4, 4]
    attention_depth = [2, 2]
    emb_dropout = [0.5, 0.5]

    use_abs_pos_emb = [True, True]
    rel_pos_bias = [False, False]
    scaled_sinu_pos_emb = [True, True]
    post_emb_norm = [False, True]

    weight_decay = [1e-6, 1e-6]
    max_epochs = [200, 200]
    attention_dim_base = [64, 64]
    optimizer_name = ["adamw", "adamw"]

    num_ensemble = len(note)

    output_dir = "/data/project/rw/projects/pandas_exam/final_posemb_modelreload/models"

    models = []
    exp_ids = []
    for idx in range(num_ensemble):
        exp_id = \
        f"TH{threshold_corr:.2f}-"\
        f"LR{lr[idx]}-"\
        f"WD{weight_decay[idx]}-"\
        f"BS{batch_size[idx]}-"\
        f"EP{max_epochs[idx]}-"\
        f"HEAD{attention_head[idx]}-"\
        f"BASE{attention_dim_base[idx]}-"\
        f"D{attention_depth[idx]}-"\
        f"DROP{emb_dropout[idx]}-"\
        f"ABSPOS{use_abs_pos_emb[idx]}-"\
        f"RELPOS{rel_pos_bias[idx]}-"\
        f"SCPOS{scaled_sinu_pos_emb[idx]}-"\
        f"NORMPOS{post_emb_norm[idx]}-"\
        f"{note[idx]}"
        exp_ids.append(exp_id)
        output_csv_path = os.path.join(output_dir, exp_id)
        model_path = os.path.join(output_csv_path, "model.pth")
        state_dict = torch.load(model_path, map_location="cpu")
        print(f"model is loaded from {model_path}")

        # Get input-dim and seq_length from dataset
        dim_in = val_loader.dataset.feature_dim
        # Num. of attention dim should be divided by num. of attention_head without residual.
        attention_dim = attention_dim_base[idx] * attention_head[idx]
        max_seq_len = val_loader.dataset.seq_length
        # Get model, optimizer, amp-scaler, and lr_scheduler
        model, _, _, _ = get_train_model(
            dim_in=dim_in,
            dim_out=2, # binary-classification
            attention_dim=attention_dim,
            attention_depth=attention_depth[idx],
            attention_head=attention_head[idx],
            max_seq_len=max_seq_len,
            emb_dropout=emb_dropout[idx],
            rel_pos_bias=rel_pos_bias[idx],
            use_abs_pos_emb=use_abs_pos_emb[idx],
            scaled_sinu_pos_emb=scaled_sinu_pos_emb[idx],
            post_emb_norm=post_emb_norm[idx],
            lr=lr[idx],
            max_epochs=max_epochs[idx],
            optimizer_name=optimizer_name[idx], 
            amp=amp,
            device=device
        )
        model.load_state_dict(state_dict)
        model.to(device)
        models.append(model)

    # Validating
    accuracy, loss = validate_ensemble(val_loader, models, device=device, amp=amp)

    output_csv_path = os.path.join(output_dir, exp_ids[0])
    output_csv_filename = os.path.join(
        output_csv_path, exp_ids[0] + f"_ensemble_{accuracy*100:.4f}.csv"
    )
    import pdb; pdb.set_trace()
    testing_ensemble(test_loader, models, output_filename=output_csv_filename, device=device, amp=amp)

    
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
