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
            # Ensembling
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
            # Ensembling
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


    #threshold_corr = 0.0
    threshold_corr = 0.1
    rnd_seed = 0
    batch_size = [128, 256]

    # Get data info.
    data_info_dict, useless_column_name, all_column_name = get_data_info(
        train_data, threshold_corr=threshold_corr
    )
    # Add useless columns
    drop_column_name = ["newID", "logging_timestamp"] + useless_column_name
    # Get survival columns for visualization
    set_all_column = set(all_column_name)
    set_drop_column_name = set(drop_column_name)
    print(f"Survived columns (th: {threshold_corr}): {set_all_column - set_drop_column_name}")
    print(f"len(drop_column_name): {len(drop_column_name)}")

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
    # Check the keys in _train_data and _val_data are disjoint.
    keys_train = set(train_data.keys())
    keys_val = set(val_data.keys())
    is_disjoint = len(keys_train.intersection(keys_val)) == 0
    if is_disjoint:
        print("The train_data and val_data are disjoint each other")
    else:
        assert 0, \
            f"keys_train.intersection(keys_val): {keys_train.intersection(keys_val)}"

    val_loader = build_dataloader(
        val_data, batch_size=batch_size[0], mode="val"
    )
    test_loader = build_dataloader(
        test_data, batch_size=batch_size[0], mode="test"
    )

    """
    # ACC: 0.9713601469993591, LOSS: 0.15020430088043213
    #TH0.00-LR1e-05-WD1e-07-BS128-EP1600-HEAD4-BASE64-D2-DROP0.5-ABSPOSTrue-RELPOSFalse-SCPOSFalse-NORMPOSFalse-final_posemb_dropcolumnbugfix_bs128_ep1600
    #TH0.00-LR1e-05-WD1e-07-BS256-EP3200-HEAD4-BASE64-D2-DROP0.5-ABSPOSTrue-RELPOSFalse-SCPOSFalse-NORMPOSTrue-final_posemb_dropcolumnbugfix_bs256_ep3200
    """

    """
    ACC: 0.9752873778343201, LOSS: 0.1402411013841629
    #TH0.10-LR1e-05-WD1e-07-BS128-EP1600-HEAD4-BASE64-D2-DROP0.5-ABSPOSTrue-RELPOSFalse-SCPOSFalse-NORMPOSFalse-final_posemb_dropcolumnbugfix_bs128_ep1600
    #TH0.10-LR1e-05-WD1e-07-BS256-EP3200-HEAD4-BASE64-D2-DROP0.5-ABSPOSTrue-RELPOSFalse-SCPOSFalse-NORMPOSTrue-final_posemb_dropcolumnbugfix_bs256_ep3200
    """

    note = ["final_posemb_dropcolumnbugfix_bs128_ep1600", "final_posemb_dropcolumnbugfix_bs256_ep3200"]
    lr = [0.00001, 0.00001]
    attention_head = [4, 4]
    attention_depth = [2, 2]
    emb_dropout = [0.5, 0.5]

    use_abs_pos_emb = [True, True]
    rel_pos_bias = [False, False]
    scaled_sinu_pos_emb = [False, False]
    post_emb_norm = [False, True]

    weight_decay = [1e-7, 1e-7]
    max_epochs = [1600, 3200]
    attention_dim_base = [64, 64]
    optimizer_name = ["adamw", "adamw"]

    num_ensemble = len(note)

    output_dir = "/data/project/rw/projects/pandas_exam/final_ensemble/models"

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

    print(f"ACC: {accuracy}, LOSS: {loss}", flush=True)

    output_csv_path = os.path.join(output_dir, exp_ids[0])
    output_csv_filename = os.path.join(
        output_csv_path, exp_ids[0] + f"_ensemble_{accuracy*100:.4f}.csv"
    )
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

    # Do ensembling
    main(train_data, test_data, drop_column_name, rnd_seed)
