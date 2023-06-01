from typing import Dict, List, Callable

import pandas as pd


def exp_grid_search(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
    fn_run_experiment: Callable,
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.0, 0.1, 0.2, 0.3]
    list_lr = [0.1, 0.001]
    list_batch_size = [128, 256]
    list_attention_head = [2, 4]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.0, 0.5]

    best_acc = 0.0
    best_loss = 10000.

    # Grid Search - TODO: Have to be re-factored.
    for emb_dropout in list_emb_dropout:
        for lr in list_lr:
            for batch_size in list_batch_size:
                for threshold_corr in list_threshold_corr:
                    accuracy, loss, drop_column_name = fn_run_experiment(
                        train_data, test_data,
                        drop_column_name,
                        threshold_corr=threshold_corr,
                        lr=lr,
                        weight_decay=1e-6,
                        batch_size=batch_size,
                        max_epochs=100,
                        attention_head=4, 
                        attention_dim_base=64,
                        attention_depth=2,
                        emb_dropout=emb_dropout,
                        device=device,
                        amp=amp
                    )

                    if best_acc < accuracy:
                        best_acc = accuracy
                        best_loss = loss
                        print(f"drop_column_name: {drop_column_name}")
                        print(f"emb_dropout: {emb_dropout}")
                        print(f"lr: {lr}")
                        print(f"batch_size: {batch_size}")
                        print(f"threshold_corr: {threshold_corr}")


def exp_grid_search_full(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
    fn_run_experiment: Callable,
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.0, 0.1, 0.2, 0.3]
    list_lr = [0.1, 0.001]
    list_batch_size = [128, 256]
    list_attention_head = [2, 4]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.0, 0.5]

    best_acc = 0.0
    best_loss = 10000.

    # Grid Search - TODO: Have to be re-factored.
    for emb_dropout in list_emb_dropout:
        for lr in list_lr:
            for batch_size in list_batch_size:
                for threshold_corr in list_threshold_corr:
                    for attention_depth in list_attention_depth:
                        for attention_head in list_attention_head:
                            accuracy, loss, drop_column_name = fn_run_experiment(
                                train_data, test_data,
                                drop_column_name,
                                threshold_corr=threshold_corr,
                                lr=lr,
                                weight_decay=1e-6,
                                batch_size=batch_size,
                                max_epochs=100,
                                attention_head=attention_head, 
                                attention_dim_base=64,
                                attention_depth=attention_depth,
                                emb_dropout=emb_dropout,
                                device=device,
                                amp=amp
                            )

                            if best_acc < accuracy:
                                best_acc = accuracy
                                best_loss = loss
                                print(f"drop_column_name: {drop_column_name}")
                                print(f"emb_dropout: {emb_dropout}")
                                print(f"lr: {lr}")
                                print(f"batch_size: {batch_size}")
                                print(f"threshold_corr: {threshold_corr}")
                                print(f"attention_head: {attention_head}")
                                print(f"attention_depth: {attention_depth}")
