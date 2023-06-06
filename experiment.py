from typing import Dict, List, Callable

import pandas as pd


def exp_grid_search(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.0, 0.1, 0.2, 0.3]
    list_lr = [0.1, 0.001]
    list_batch_size = [128, 256]
    list_attention_head = [2, 4]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.0, 0.5]
    weight_decay = 1e-6
    max_epochs = 100
    optimizer_name = "adamw"
    attention_head = 4
    attention_dim_base = 64
    attention_depth = 2

    best_acc = 0.0
    best_loss = 10000.

    # Grid Search - TODO: Have to be re-factored.
    for emb_dropout in list_emb_dropout:
        for lr in list_lr:
            for batch_size in list_batch_size:
                for threshold_corr in list_threshold_corr:
                    exp_id = \
                        f"TH{threshold_corr:.2f}-"\
                        f"LR{lr}-"\
                        f"WD{weight_decay}-"\
                        f"BS{batch_size}-"\
                        f"EP{max_epochs}-"\
                        f"OPT{optimizer_name}-"\
                        f"HEAD{attention_head}-"\
                        f"BASE{attention_dim_base}-"\
                        f"D{attention_depth}-"\
                        f"DROP{emb_dropout}-"\
                        f"{note}"
                    accuracy, loss  = fn_run_experiment(
                        train_data, test_data,
                        exp_id=exp_id,
                        threshold_corr=threshold_corr,
                        lr=lr,
                        weight_decay=weight_decay,
                        batch_size=batch_size,
                        max_epochs=max_epochs,
                        optimizer_name=optimizer_name,
                        attention_head=attention_head, 
                        attention_dim_base=attention_dim_base,
                        attention_depth=attention_depth,
                        emb_dropout=emb_dropout,
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
                        print(f"optimizer_name: {optimizer_name}")
                        print(f"threshold_corr: {threshold_corr}")


def exp_grid_search_full(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    #list_threshold_corr = [0.0, 0.1, 0.2, 0.3]
    list_threshold_corr = [0.4, 0.35, 0.3, 0.2, 0.1, 0.0]
    list_lr = [0.1, 0.001]
    list_batch_size = [128, 256]
    list_attention_head = [2, 4]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.0, 0.5]
    weight_decay = 1e-6
    max_epochs = 100
    attention_dim_base = 64

    best_acc = 0.0
    best_loss = 10000.

    # Grid Search - TODO: Have to be re-factored.
    for emb_dropout in list_emb_dropout:
        for lr in list_lr:
            for batch_size in list_batch_size:
                for attention_depth in list_attention_depth:
                    for attention_head in list_attention_head:
                        for threshold_corr in list_threshold_corr:
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
                                f"{note}"
                            accuracy, loss = fn_run_experiment(
                                train_data, test_data,
                                exp_id=exp_id,
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


def exp_grid_search_full_bs32(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    list_lr = [0.00005, 0.0001, 0.0002, 0.001, 0.01]
    list_batch_size = [32, 64, 128]
    list_attention_head = [2, 4]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5, 0.2, 0.0]

    weight_decay = 1e-6
    max_epochs = 200
    attention_dim_base = 64

    best_acc = 0.0
    best_loss = 10000.

    # Grid Search - TODO: Have to be re-factored.
    for emb_dropout in list_emb_dropout:
        for lr in list_lr:
            for batch_size in list_batch_size:
                for attention_depth in list_attention_depth:
                    for attention_head in list_attention_head:
                        for threshold_corr in list_threshold_corr:
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
                                f"{note}"
                            accuracy, loss = fn_run_experiment(
                                train_data, test_data,
                                exp_id=exp_id,
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



def exp_grid_search_full_bs1632_head68_reprod(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.25]
    list_lr = [0.00001]
    list_batch_size = [16]
    list_attention_head = [4]
    list_attention_depth = [2]
    list_emb_dropout = [0.5]

    use_abs_pos_emb = True
    rel_pos_bias = False
    post_emb_norm = False
    scaled_sinu_pos_emb = True

    weight_decay = 1e-6
    max_epochs = 200
    attention_dim_base = 64

    best_acc = 0.0
    best_loss = 10000.

    # Grid Search - TODO: Have to be re-factored.
    for emb_dropout in list_emb_dropout:
        for lr in list_lr:
            for batch_size in list_batch_size:
                for attention_depth in list_attention_depth:
                    for attention_head in list_attention_head:
                        for threshold_corr in list_threshold_corr:
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
                                f"{note}"
                            accuracy, loss = fn_run_experiment(
                                train_data, test_data,
                                exp_id=exp_id,
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
                                rel_pos_bias=rel_pos_bias,
                                use_abs_pos_emb=use_abs_pos_emb,
                                post_emb_norm=post_emb_norm,
                                scaled_sinu_pos_emb=scaled_sinu_pos_emb,
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





def exp_grid_search_full_bs1632_head68(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    list_lr = [0.00005, 0.0001, 0.0002, 0.001, 0.01]
    list_batch_size = [16, 32, 64]
    list_attention_head = [6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5, 0.2, 0.0]

    rel_pos_bias = True
    use_abs_pos_emb = True
    post_emb_norm = False
    scaled_sinu_pos_emb = False

    weight_decay = 1e-6
    max_epochs = 200
    attention_dim_base = 64

    best_acc = 0.0
    best_loss = 10000.

    # Grid Search - TODO: Have to be re-factored.
    for emb_dropout in list_emb_dropout:
        for lr in list_lr:
            for batch_size in list_batch_size:
                for attention_depth in list_attention_depth:
                    for attention_head in list_attention_head:
                        for threshold_corr in list_threshold_corr:
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
                                f"{note}"
                            accuracy, loss = fn_run_experiment(
                                train_data, test_data,
                                exp_id=exp_id,
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
                                rel_pos_bias=rel_pos_bias,
                                use_abs_pos_emb=use_abs_pos_emb,
                                post_emb_norm=post_emb_norm,
                                scaled_sinu_pos_emb=scaled_sinu_pos_emb,
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


def exp_grid_search_full_bs1632_head68_posembed(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    rnd_seed: int,
    output_dir: str,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.35, 0.3, 0.25, 0.2]
    list_lr = [0.00001, 0.00005, 0.0001]
    list_batch_size = [16, 32, 64]
    list_attention_head = [4, 6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5]
    list_weight_decay = [1e-6, 1e-7]
    #rel_pos_bias = [True] # Set always be True
    #scaled_sinu_pos_emb = [True, False]
    #use_abs_pos_emb = [True] # Set always be True
    #post_emb_norm = [True, False]
    list_pos_emb_config = [
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
    ]

    weight_decay = 1e-6
    max_epochs = 200
    attention_dim_base = 64

    best_acc = 0.0
    best_loss = 10000.

    # Grid Search - TODO: Have to be re-factored.
    for emb_dropout in list_emb_dropout:
        for lr in list_lr:
            for batch_size in list_batch_size:
                for attention_depth in list_attention_depth:
                    for attention_head in list_attention_head:
                        for threshold_corr in list_threshold_corr:
                            for pos_emb_config in list_pos_emb_config:
                                rel_pos_bias = pos_emb_config["rel_pos_bias"]
                                use_abs_pos_emb = pos_emb_config["use_abs_pos_emb"]
                                scaled_sinu_pos_emb = pos_emb_config["scaled_sinu_pos_emb"]
                                post_emb_norm = pos_emb_config["post_emb_norm"]
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


def exp_grid_search_full_bs1632_head68_posembed_eventdrop(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    rnd_seed: int,
    output_dir: str,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.35, 0.3, 0.25, 0.2]
    list_lr = [0.00001, 0.00005, 0.0001]
    list_batch_size = [16, 32, 64]
    list_attention_head = [4, 6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5]
    list_event_drop = [0.2]
    list_weight_decay = [1e-6, 1e-7]
    list_pos_emb_config = [
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
    ]

    weight_decay = 1e-6
    max_epochs = 200
    attention_dim_base = 64

    best_acc = 0.0
    best_loss = 10000.

    # Grid Search - TODO: Have to be re-factored.
    for lr in list_lr:
        for batch_size in list_batch_size:
            for attention_depth in list_attention_depth:
                for attention_head in list_attention_head:
                    for threshold_corr in list_threshold_corr:
                        for pos_emb_config in list_pos_emb_config:
                            for event_drop in list_event_drop:
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
                                    f"EVENTDROP{event_drop}-"\
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
                                    event_drop=event_drop,
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












def exp_grid_search_full_bs32_wd1e_7(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    list_lr = [0.00005, 0.0001, 0.0002, 0.001, 0.01]
    list_batch_size = [32, 64, 128]
    list_attention_head = [2, 4]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5, 0.2, 0.0]

    weight_decay = 1e-7
    max_epochs = 200
    attention_dim_base = 64

    best_acc = 0.0
    best_loss = 10000.

    # Grid Search - TODO: Have to be re-factored.
    for emb_dropout in list_emb_dropout:
        for lr in list_lr:
            for batch_size in list_batch_size:
                for attention_depth in list_attention_depth:
                    for attention_head in list_attention_head:
                        for threshold_corr in list_threshold_corr:
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
                                f"{note}"
                            accuracy, loss = fn_run_experiment(
                                train_data, test_data,
                                exp_id=exp_id,
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


def exp_grid_search_full_bs32_wd1e_7_head123(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    list_lr = [0.00005, 0.0001, 0.0002, 0.001, 0.01]
    list_batch_size = [32, 64, 128]
    list_attention_head = [1, 2, 3]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5, 0.2, 0.0]

    weight_decay = 1e-7
    max_epochs = 200
    attention_dim_base = 64

    best_acc = 0.0
    best_loss = 10000.

    # Grid Search - TODO: Have to be re-factored.
    for emb_dropout in list_emb_dropout:
        for lr in list_lr:
            for batch_size in list_batch_size:
                for attention_depth in list_attention_depth:
                    for attention_head in list_attention_head:
                        for threshold_corr in list_threshold_corr:
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
                                f"{note}"
                            accuracy, loss = fn_run_experiment(
                                train_data, test_data,
                                exp_id=exp_id,
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


def exp_grid_search_full_bs32_wd1e_7_head123_optname(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.35, 0.3, 0.25, 0.2]
    list_lr = [0.00005, 0.0001, 0.0002, 0.001, 0.01]
    list_batch_size = [32, 64]
    list_attention_head = [3, 4, 5]
    list_attention_depth = [1, 2, 3]
    list_emb_dropout = [0.5, 0.2, 0.0]
    list_optimizer_name = ["adam", "sgd", "adamw"]

    weight_decay = 1e-7
    max_epochs = 200
    attention_dim_base = 64

    best_acc = 0.0
    best_loss = 10000.

    # Grid Search - TODO: Have to be re-factored.
    for emb_dropout in list_emb_dropout:
        for optimizer_name in list_optimizer_name:
            for lr in list_lr:
                for batch_size in list_batch_size:
                    for attention_depth in list_attention_depth:
                        for attention_head in list_attention_head:
                            for threshold_corr in list_threshold_corr:
                                exp_id = \
                                    f"TH{threshold_corr:.2f}-"\
                                    f"LR{lr}-"\
                                    f"WD{weight_decay}-"\
                                    f"BS{batch_size}-"\
                                    f"EP{max_epochs}-"\
                                    f"OPT{optimizer_name}-"\
                                    f"HEAD{attention_head}-"\
                                    f"BASE{attention_dim_base}-"\
                                    f"D{attention_depth}-"\
                                    f"DROP{emb_dropout}-"\
                                    f"{note}"
                                accuracy, loss = fn_run_experiment(
                                    train_data, test_data,
                                    exp_id=exp_id,
                                    threshold_corr=threshold_corr,
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    batch_size=batch_size,
                                    max_epochs=max_epochs,
                                    optimizer_name=optimizer_name,
                                    attention_head=attention_head, 
                                    attention_dim_base=attention_dim_base,
                                    attention_depth=attention_depth,
                                    emb_dropout=emb_dropout,
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
                                    print(f"optimizer_name: {optimizer_name}")
                                    print(f"threshold_corr: {threshold_corr}")
                                    print(f"attention_head: {attention_head}")
                                    print(f"attention_depth: {attention_depth}")


def exp_grid_search_full_bs1632_head68_posembed_final_ep250(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    rnd_seed: int,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.5, 0.0]
    list_lr = [0.00001, 0.00005, 0.0001]
    list_batch_size = [64, 32, 16]
    list_attention_head = [4, 6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5]
    list_weight_decay = [1e-6, 1e-7]
    list_pos_emb_config = [
        {"use_abs_pos_emb": True, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
    ]

    max_epochs = 250
    attention_dim_base = 64

    best_acc = 0.0
    best_loss = 10000.

    # Grid Search - TODO: Have to be re-factored.
    for emb_dropout in list_emb_dropout:
        for weight_decay in list_weight_decay:
            for lr in list_lr:
                for batch_size in list_batch_size:
                    for attention_depth in list_attention_depth:
                        for attention_head in list_attention_head:
                            for threshold_corr in list_threshold_corr:
                                for pos_emb_config in list_pos_emb_config:
                                    rel_pos_bias = pos_emb_config["rel_pos_bias"]
                                    use_abs_pos_emb = pos_emb_config["use_abs_pos_emb"]
                                    scaled_sinu_pos_emb = pos_emb_config["scaled_sinu_pos_emb"]
                                    post_emb_norm = pos_emb_config["post_emb_norm"]
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


def exp_dev(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    rnd_seed: int,
    output_dir: str,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.35, 0.3, 0.25, 0.2]
    list_lr = [0.00001]
    list_batch_size = [16]
    list_attention_head = [4]
    list_attention_depth = [2]
    list_emb_dropout = [0.5]
    list_event_drop = [0.2]
    list_weight_decay = [1e-6]
    list_pos_emb_config = [
        {"use_abs_pos_emb": True, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
    ]

    weight_decay = 1e-6
    max_epochs = 200
    attention_dim_base = 64

    best_acc = 0.0
    best_loss = 10000.

    # Grid Search - TODO: Have to be re-factored.
    for lr in list_lr:
        for batch_size in list_batch_size:
            for attention_depth in list_attention_depth:
                for attention_head in list_attention_head:
                    for threshold_corr in list_threshold_corr:
                        for pos_emb_config in list_pos_emb_config:
                            for event_drop in list_event_drop:
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
                                    f"EVENTDROP{event_drop}-"\
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
                                    event_drop=event_drop,
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


def exp_grid_search_full_bs1632_head68_posembed_renew(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    rnd_seed: int,
    output_dir: str,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.35, 0.3, 0.25, 0.2]
    list_lr = [0.00001, 0.00005, 0.0001]
    list_batch_size = [16, 32]
    list_attention_head = [4, 6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5]
    list_weight_decay = [1e-7, 1e-6]
    #rel_pos_bias = [True] # Set always be True
    #scaled_sinu_pos_emb = [True, False]
    #use_abs_pos_emb = [True] # Set always be True
    #post_emb_norm = [True, False]
    list_pos_emb_config = [
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": False, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": False},
        {"use_abs_pos_emb": False, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": True},
    ]

    max_epochs = 200
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


def exp_grid_search_full_bs3264(
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
    list_batch_size = [32, 64]
    list_attention_head = [4, 6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5]
    list_weight_decay = [1e-7, 1e-6]
    #rel_pos_bias = [True] # Set always be True
    #scaled_sinu_pos_emb = [True, False]
    #use_abs_pos_emb = [True] # Set always be True
    #post_emb_norm = [True, False]
    list_pos_emb_config = [
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": True},
    ]

    max_epochs = 200
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


def exp_grid_search_full_bs3264_ep250(
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
    list_batch_size = [32, 64]
    list_attention_head = [4, 6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5]
    list_weight_decay = [1e-7, 1e-6]
    #rel_pos_bias = [True] # Set always be True
    #scaled_sinu_pos_emb = [True, False]
    #use_abs_pos_emb = [True] # Set always be True
    #post_emb_norm = [True, False]
    list_pos_emb_config = [
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": True},
    ]

    max_epochs = 250
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


def exp_grid_search_full_bs3264_ep300(
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
    list_batch_size = [32, 64, 16]
    list_attention_head = [4, 6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5]
    list_weight_decay = [1e-7, 1e-6]
    #rel_pos_bias = [True] # Set always be True
    #scaled_sinu_pos_emb = [True, False]
    #use_abs_pos_emb = [True] # Set always be True
    #post_emb_norm = [True, False]
    list_pos_emb_config = [
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": True},
    ]

    max_epochs = 300
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


def exp_grid_search_full_bs3264_ep300_eventdrop(
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
    list_batch_size = [32, 64, 16]
    list_attention_head = [4, 6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5]
    list_event_drop = [0.2]
    list_weight_decay = [1e-7, 1e-6]
    #rel_pos_bias = [True] # Set always be True
    #scaled_sinu_pos_emb = [True, False]
    #use_abs_pos_emb = [True] # Set always be True
    #post_emb_norm = [True, False]
    list_pos_emb_config = [
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": True},
    ]

    max_epochs = 300
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
                                event_drop = list_event_drop[0]
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
                                    f"EVENTDROP{event_drop}-"\
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
                                    event_drop=event_drop,
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

def exp_grid_search_full_bs3264_ep400(
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
    list_batch_size = [32, 64, 16]
    list_attention_head = [4, 6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5]
    list_weight_decay = [1e-7, 1e-6]
    #rel_pos_bias = [True] # Set always be True
    #scaled_sinu_pos_emb = [True, False]
    #use_abs_pos_emb = [True] # Set always be True
    #post_emb_norm = [True, False]
    list_pos_emb_config = [
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": True},
    ]

    max_epochs = 400
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


def exp_grid_search_full_bs3264_ep500(
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
    list_batch_size = [32, 64, 16]
    list_attention_head = [4, 6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5]
    list_weight_decay = [1e-7, 1e-6]
    #rel_pos_bias = [True] # Set always be True
    #scaled_sinu_pos_emb = [True, False]
    #use_abs_pos_emb = [True] # Set always be True
    #post_emb_norm = [True, False]
    list_pos_emb_config = [
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": True},
    ]

    max_epochs = 500
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


def exp_grid_search_full_bs3264_ep600(
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
    list_batch_size = [32, 64, 16]
    list_attention_head = [4, 6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5]
    list_weight_decay = [1e-7, 1e-6]
    #rel_pos_bias = [True] # Set always be True
    #scaled_sinu_pos_emb = [True, False]
    #use_abs_pos_emb = [True] # Set always be True
    #post_emb_norm = [True, False]
    list_pos_emb_config = [
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": True},
    ]

    max_epochs = 600
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


def exp_grid_search_full_bs64_ep800_reverse(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    rnd_seed: int,
    output_dir: str,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    #list_threshold_corr = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35]
    list_threshold_corr = [0.35, 0.3, 0.25, 0.2, 0.1, 0.0]
    list_lr = [0.00001, 0.00005, 0.0001]
    list_batch_size = [64]
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

    max_epochs = 800
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





def exp_grid_search_full_bs64_ep800(
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
    list_batch_size = [64]
    list_attention_head = [4, 6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5]
    list_weight_decay = [1e-7, 1e-6]
    #rel_pos_bias = [True] # Set always be True
    #scaled_sinu_pos_emb = [True, False]
    #use_abs_pos_emb = [True] # Set always be True
    #post_emb_norm = [True, False]
    list_pos_emb_config = [
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
        {"use_abs_pos_emb": False, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": False},
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": True, "post_emb_norm": True},
    ]

    max_epochs = 800
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


def exp_grid_search_full_bs256_ep3200(
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
    list_batch_size = [256]
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

    max_epochs = 3200
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


def exp_grid_search_full_bs256_ep3200_onepoint(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fn_run_experiment: Callable,
    rnd_seed: int,
    output_dir: str,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.1]
    list_lr = [0.00001]
    list_batch_size = [256]
    list_attention_head = [4, 6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5]
    list_weight_decay = [1e-7]
    #rel_pos_bias = [True] # Set always be True
    #scaled_sinu_pos_emb = [True, False]
    #use_abs_pos_emb = [True] # Set always be True
    #post_emb_norm = [True, False]
    list_pos_emb_config = [
        {"use_abs_pos_emb": True, "rel_pos_bias": False, "scaled_sinu_pos_emb": False, "post_emb_norm": False},
    ]

    max_epochs = 3200
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


