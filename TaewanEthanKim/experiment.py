from typing import Dict, List, Callable

import pandas as pd


def exp_grid_search_full_bs128_ep1600(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    rnd_seed: int,
    output_dir: str,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35]
    list_lr = [0.00001, 0.00005, 0.0001]
    list_batch_size = [128]
    list_attention_head = [4, 6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5]
    list_weight_decay = [1e-7, 1e-6]
    #rel_pos_bias = [True] # Set always be True
    #scaled_sinu_pos_emb = [True, False]
    #use_abs_pos_emb = [True] # Set always be True
    #post_emb_norm = [True, False]
    list_pos_emb_config = [
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": False},
    ]

    max_epochs = 1600
    attention_dim_base = 64

    best_acc = 0.0
    best_loss = 10000.

    # Grid Search - TODO: Have to be re-factored.
    for weight_decay in list_weight_decay:
        for lr in list_lr:
            for batch_size in list_batch_size:
                for attention_depth in list_attention_depth:
                    for attention_head in list_attention_head:
                        for pos_emb_config in list_pos_emb_config:
                            for threshold_corr in list_threshold_corr:
                                rel_pos_bias = pos_emb_config["rel_pos_bias"]
                                use_abs_pos_emb = pos_emb_config["use_abs_pos_emb"]
                                scaled_sinu_pos_emb = pos_emb_config["scaled_sinu_pos_emb"]
                                post_emb_norm = pos_emb_config["post_emb_norm"]
                                emb_dropout = list_emb_dropout[0]
                                exp_id = \
                                    f"TH{threshold_corr:.2f}-"\
                                    f"LR{lr}-"\
                                    f"WD{weight_decay}-"\
                                    f"BS{batch_size}-"\
                                    f"EP{max_epochs}-"\
                                    f"HEAD{attention_head}-"\
                                    f"BASE{attention_dim_base}-"\
                                    f"D{attention_depth}-"\
                                    f"DROP{emb_dropout}-"\
                                    f"ABSPOS{use_abs_pos_emb}-"\
                                    f"RELPOS{rel_pos_bias}-"\
                                    f"SCPOS{scaled_sinu_pos_emb}-"\
                                    f"NORMPOS{post_emb_norm}-"\
                                    f"{note}"
                                accuracy, loss = fn_run_experiment(
                                    train_data, test_data,
                                    exp_id=exp_id,
                                    output_dir=output_dir,
                                    rnd_seed=rnd_seed,
                                    threshold_corr=threshold_corr,
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    batch_size=batch_size,
                                    max_epochs=max_epochs,
                                    optimizer_name="adamw",
                                    attention_head=attention_head, 
                                    attention_dim_base=attention_dim_base,
                                    attention_depth=attention_depth,
                                    emb_dropout=emb_dropout,
                                    event_drop=0.0,
                                    rel_pos_bias=rel_pos_bias,
                                    use_abs_pos_emb=use_abs_pos_emb,
                                    scaled_sinu_pos_emb=scaled_sinu_pos_emb,
                                    post_emb_norm=post_emb_norm,
                                    device=device,
                                    amp=amp
                                )

                                if best_acc < accuracy:
                                    best_acc = accuracy
                                    best_loss = loss
                                    print("BEST CONFIGURATION")
                                    print(f"best_acc: {best_acc}")
                                    print(f"best_loss: {best_loss}")
                                    print(f"emb_dropout: {emb_dropout}")
                                    print(f"lr: {lr}")
                                    print(f"batch_size: {batch_size}")
                                    print(f"threshold_corr: {threshold_corr}")
                                    print(f"attention_head: {attention_head}")
                                    print(f"attention_depth: {attention_depth}")
