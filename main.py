from typing import Tuple, List, Dict

import os

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
from utils import AverageMeter, set_random_seed, report_summary, clear_object
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


def run_train(
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    output_dir: str,
    exp_id: str,
    writer: torch.utils.tensorboard.SummaryWriter,
    lr: float = 0.00001,
    weight_decay: float = 1e-9,
    batch_size: int = 128,
    max_epochs: int = 100,
    optimizer_name: str = "adamw",
    attention_head: int = 2,
    attention_dim_base: int = 64,
    attention_depth = 2,
    rel_pos_bias: bool = True,
    use_abs_pos_emb: bool = True, 
    scaled_sinu_pos_emb: bool = False,
    post_emb_norm: bool = False,
    emb_dropout = 0.5,
    device: str = "cuda:0",
    amp: bool = False,
):

    disp_freq = 200
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

    # Num. of attention dim should be divided by num. of attention_head without residual.
    attention_dim = attention_dim_base * attention_head
    # Get input-dim and seq_length from dataset
    dim_in = train_loader.dataset.feature_dim
    max_seq_len = train_loader.dataset.seq_length
    # Get model, optimizer, amp-scaler, and lr_scheduler
    M, optimizer, scheduler, amp_scaler = get_train_model(
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

    global_iters = 0
    loss_avgmeter = AverageMeter()
    # Main loop
    for epoch in range(max_epochs):
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # inputs.shape: (B, seq_length, feature_dim)

            M.train()

            with torch.autocast("cuda", enabled=amp):
                # Forward
                logits = M(inputs)
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
                    M.parameters(), max_grad_norm
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
                report_summary(
                    writer, epoch, global_iters, "train",
                    loss=loss, grad_norm=grad_norm
                )

        # Update per epoch
        scheduler.step()

        # Validating
        accuracy, loss = validate(val_loader, M, device=device, amp=amp)
        print(
            f"ep: {epoch}, ACC: {accuracy * 100:.4f}, LOSS: {loss:.2f}",
            flush=True
        ) 
        report_summary(
            writer, epoch, global_iters, "val",
            loss=loss, accuracy=accuracy
        )

        if best_acc < accuracy:
            # Update best score
            best_acc = accuracy
            best_loss = loss
            print(
                f"Best so far: "\
                f"BEST-ACC: {best_acc * 100:.4f}, "\
                f"BEST-LOSS: {best_loss:.2f} "\
                f"in ep {epoch}",
                flush=True
            )

            # Get dir.
            output_csv_path = os.path.join(output_dir, "models", exp_id)
            os.makedirs(output_csv_path, exist_ok=True)

            # Save statedict
            state_dict = M.state_dict()
            ckpt_filename = os.path.join(output_csv_path, "model.pth")
            torch.save(state_dict, ckpt_filename)
            print(f"Save ckpt: {ckpt_filename}")

            # Get dir.
            output_csv_filename = os.path.join(
                output_csv_path, exp_id + f"_ep{epoch:03d}_{best_acc*100:.4f}.csv"
            )
            # Perform testing
            testing(
                test_loader, M,
                output_filename=output_csv_filename,
                device=device, amp=amp
            )

    clear_object(M)
    clear_object(optimizer)
    clear_object(scheduler)
    clear_object(amp_scaler)

    return best_acc, best_loss


def run_experiment(
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

    # Set output dir.
    #output_dir = "test"
    #output_dir = "results"
    #output_dir = "renew"
    #output_dir = "reborn"
    #output_dir = "reprod"
    #output_dir = "reprod_posemb"
    #output_dir = "final"
    output_dir = "final_posemb"
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(
        os.path.join(output_dir, "summary", exp_id),
        comment=exp_id
    )
    
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

    """
    # For visualizing survived features
    for threshold_corr in [0.35, 0.3, 0.25, 0.2, 0.15, 0.1]:
        data_info_dict, useless_column_name, all_column_name = get_data_info(
            train_data, threshold_corr=threshold_corr
        )
        # Add useless columns
        drop_column_name += useless_column_name
        # Get survival columns for visualization
        set_all_column = set(all_column_name)
        set_useless_column = set(useless_column_name)
        print(f"Survived columns (th: {threshold_corr}): {set_all_column - set_useless_column}") 
        import pdb; pdb.set_trace()
    """

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

    # Training
    best_acc, best_loss = run_train(
        train_data, val_data, test_data,
        output_dir=output_dir,
        exp_id=exp_id,
        writer=writer,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        optimizer_name=optimizer_name,
        attention_head=attention_head,
        attention_dim_base=attention_dim_base,
        attention_depth=attention_depth,
        emb_dropout=emb_dropout,
        rel_pos_bias=rel_pos_bias,
        use_abs_pos_emb=use_abs_pos_emb,
        scaled_sinu_pos_emb=scaled_sinu_pos_emb,
        post_emb_norm=post_emb_norm,
        device=device,
        amp=amp
    )

    # Cleansing
    clear_object(writer)
    clear_object(train_data)
    clear_object(val_data)
    clear_object(test_data)

    return best_acc, best_loss, drop_column_name


def main(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
    rnd_seed: int,
    device: str = "cuda:0",
    amp: bool = False
) -> bool:

    # Conduct experiments (grid-search)
    """
    experiment.exp_grid_search(
        train_data, test_data,
        drop_column_name,
        run_experiment,
        device=device,
        amp=amp
    )
    experiment.exp_grid_search_full(
        train_data, test_data,
        drop_column_name,
        run_experiment,
        device=device,
        amp=amp
    )
    experiment.exp_grid_search_full_epoch200(
        train_data, test_data,
        drop_column_name,
        run_experiment,
        note="",
        device=device,
        amp=amp
    )
    experiment.exp_grid_search_full_bs32(
        train_data, test_data,
        drop_column_name,
        run_experiment,
        note="saveckpt",
        device=device,
        amp=amp
    )
    experiment.exp_grid_search_full_bs32_wd1e_7_head123(
        train_data, test_data,
        drop_column_name,
        run_experiment,
        note="head123",
        device=device,
        amp=amp
    )
    experiment.exp_grid_search_full_bs1632_head68(
        train_data, test_data,
        drop_column_name,
        run_experiment,
        note="bs1632-head68",
        device=device,
        amp=amp
    )
    experiment.exp_grid_search_full_bs32_wd1e_7_head123_optname(
        train_data, test_data,
        drop_column_name,
        run_experiment,
        note="opt",
        device=device,
        amp=amp
    )
    experiment.exp_grid_search_full_bs1632_head68_posembed(
        train_data, test_data,
        drop_column_name,
        run_experiment,
        note="posembed-reprod",
        device=device,
        amp=amp
    )
    experiment.exp_grid_search_full_bs1632_head68_reprod(
        train_data, test_data,
        drop_column_name,
        run_experiment,
        note="bs1632-head68-reprod",
        device=device,
        amp=amp
    )
    """
    """
    experiment.exp_grid_search_full_bs1632_head68_posembed_final(
        train_data, test_data,
        drop_column_name,
        run_experiment,
        note="final",
        device=device,
        amp=amp
    )
    """
    experiment.exp_grid_search_full_bs1632_head68_posembed(
        train_data, test_data,
        drop_column_name,
        run_experiment,
        rnd_seed,
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
