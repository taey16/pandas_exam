from typing import Tuple

import numpy as np
#import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm

import torch

from dataset.dataset import (
    read_csv,
    group_by_newID,
    shuffle_by_key,
    train_val_split,
    train_val_split_by_key,
    print_data_info,
    build_dataloader,
    normalize_data,
    check_data
)

from models.model import get_train_model
from utils import AverageMeter, set_random_seed


@torch.no_grad()
def validate(loader, model, device: str = "cuda:0", amp: bool = True):
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
            loss += torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            ).sum()
            
            pred = torch.softmax(logits, dim=2)
            pred = torch.argmax(pred, dim=2)

            # onehot to label again
            targets = torch.argmax(targets, dim=2)
            correct += (pred == targets).sum()
            sample_counter += batch_size * seq_length

    accuracy = float(correct / sample_counter)
    loss = float(loss / sample_counter)

    return accuracy, loss


def run_train(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    device: str = "cuda:0",
    amp: bool = False,
):

    disp_freq = 50
    max_grad_norm = 5.0
    max_epochs = 100
    lr = 0.00001
    batch_size = 128
    amp = True

    #train_data = normalize_data(train_data, method="minmax")
    #val_data = normalize_data(val_data, method="minmax")
    #test_data = normalize_data(test_data, method="minmax")
    #check_data(train_data)
    #check_data(val_data)
    #check_data(test_data)

    train_loader = build_dataloader(train_data, batch_size=batch_size, mode="train")
    val_loader = build_dataloader(val_data, batch_size=batch_size, mode="val")
    test_loader = build_dataloader(test_data, batch_size=batch_size, mode="test")

    dim_in = train_loader.dataset.feature_dim
    dim_out = 2
    max_seq_len = train_loader.dataset.seq_length
    emb_dropout = 0.5
    attention_head = 2
    attention_dim = 64 * attention_head
    attention_depth = 2
    assert attention_dim % attention_head == 0
    model, optimizer, scheduler, amp_scaler = get_train_model(
        dim_in=dim_in,
        dim_out=dim_out,
        attention_dim=attention_dim,
        attention_depth=attention_depth,
        attention_head=attention_head,
        max_seq_len=max_seq_len,
        lr=lr,
        max_epochs=max_epochs,
        amp=amp,
        device=device
    )

    global_iters = 0
    loss_avgmeter = AverageMeter()
    for epoch in range(max_epochs):
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # inputs.shape: (B, seq_length, feature_dim)

            #inputs[0] = torch.nn.ConstantPad1d((0, max_seq_len - inputs[0].shape[0]), 0)(inputs[0])
            ## pad all seqs to desired length
            #inputs = torch.nn.utils.rnn.pad_sequence(inputs)

            with torch.autocast("cuda", enabled=amp):
                logits = model(inputs)
                #import pdb; pdb.set_trace()
                #print(logits)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="mean"
                )

            if amp_scaler is not None:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
            else:
                loss.backward()

            if max_grad_norm > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
            else:
                grad_norm = -1.0

            if amp_scaler is not None:
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                optimizer.step()

            global_iters += 1

            if idx % disp_freq == 0:
                num_neg = int(targets[:,:,0].sum()) / 10
                num_pos = int(targets[:,:,1].sum()) / 10
                assert num_neg + num_pos == batch_size
                loss = float(loss)
                loss_avgmeter.update(loss)
                grad_norm = float(grad_norm)
                amp_scale = float(amp_scaler.get_scale()) if amp_scaler is not None else 0
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"ep: {epoch} it: {global_iters} "\
                    f"loss: {loss:.2f}({loss_avgmeter.avg:.7f}), "\
                    f"g: {grad_norm:.4f}, s: {amp_scale}, lr: {lr} "\
                    f"#pos: {num_pos}, #neg: {num_neg}"
                )

        # Update per epoch
        scheduler.step()

        # Validating
        accuracy, loss = validate(val_loader, model, device=device, amp=amp) 
        print(
            f"ep: {epoch}, ACC: {accuracy * 100:.4f}, LOSS: {loss:.2f}"
        )
        

if __name__ == "__main__":
    set_random_seed(0)

    train_csv = "inputs/abusingDetectionTrainDataset.csv"
    test_csv = "inputs/abusingDetectionTestDataset.csv"

    drop_column_name = ["newID", "logging_timestamp"]
    """
    drop_column_name = [
        "newID", "char_jobcode", "char_level", "logging_timestamp",
        "charStatA", "charStatB", "charStatC", "charStatD", "charStatE", "charStatF", "charStatG",
        "socialAmountA", "socialBooleanA", "socialBooleanB",
        "accountMetaAmountA",
    ]
    """

    train_data = read_csv(train_csv)
    test_data = read_csv(test_csv)

    min_max_dict, useless_column_name = print_data_info(train_data)

    drop_column_name += useless_column_name

    train_data = group_by_newID(train_data, min_max_dict=min_max_dict, drop_label_name=drop_column_name, mode="train")
    test_data = group_by_newID(test_data, min_max_dict=min_max_dict, drop_label_name=drop_column_name, mode="test")

    train_data = shuffle_by_key(train_data)

    train_data, val_data = train_val_split_by_key(train_data)

    run_train(train_data, val_data, test_data)
