from typing import Dict, List, Callable

import pandas as pd


def infer(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
    fn_run_infer: Callable,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    # TH0.25-LR1e-05-WD1e-06-BS16-EP200-HEAD4-BASE64-D2-DROP0.5-ABSPOSTrue-RELPOSFalse-SCPOSFalse-NORMPOSTrue-posembed

    #TH0.30-LR5e-05-WD1e-06-BS32-EP200-HEAD6-BASE64-D2-DROP0.5-bs1632-head68

    #TH0.35-LR5e-05-WD1e-06-BS16-EP200-HEAD6-BASE64-D2-DROP0.5-bs1632-head68

    # TH0.35-LR1e-05-WD1e-06-BS16-EP200-HEAD6-BASE64-D2-DROP0.5-ABSPOSTrue-RELPOSTrue-SCPOSFalse-NORMPOSTrue-posembed

    note = "posembed"
    list_threshold_corr = [0.35]
    list_lr = [0.00001]
    list_batch_size = [16]
    list_attention_head = [6]
    list_attention_depth = [2]
    list_emb_dropout = [0.5]

    use_abs_pos_emb = True
    rel_pos_bias = True
    scaled_sinu_pos_emb = False
    post_emb_norm = True

    weight_decay = 1e-6
    max_epochs = 200
    attention_dim_base = 64
    optimizer_name = "adamw"

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
                                f"ABSPOS{use_abs_pos_emb}-"\
                                f"RELPOS{rel_pos_bias}-"\
                                f"SCPOS{scaled_sinu_pos_emb}-"\
                                f"NORMPOS{post_emb_norm}-"\
                                f"{note}"
                            accuracy, loss, drop_column_name = fn_run_infer(
                                train_data, test_data,
                                drop_column_name,
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
                                rel_pos_bias=rel_pos_bias,
                                use_abs_pos_emb=use_abs_pos_emb,
                                scaled_sinu_pos_emb=scaled_sinu_pos_emb,
                                post_emb_norm=post_emb_norm,
                                device=device,
                                amp=amp
                            )



def exp_grid_search(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
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
                    accuracy, loss, drop_column_name = fn_run_experiment(
                        train_data, test_data,
                        drop_column_name,
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
                        print(f"drop_column_name: {drop_column_name}")
                        print(f"emb_dropout: {emb_dropout}")
                        print(f"lr: {lr}")
                        print(f"batch_size: {batch_size}")
                        print(f"optimizer_name: {optimizer_name}")
                        print(f"threshold_corr: {threshold_corr}")


def exp_grid_search_full(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
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
                            accuracy, loss, drop_column_name = fn_run_experiment(
                                train_data, test_data,
                                drop_column_name,
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
                                print(f"drop_column_name: {drop_column_name}")
                                print(f"emb_dropout: {emb_dropout}")
                                print(f"lr: {lr}")
                                print(f"batch_size: {batch_size}")
                                print(f"threshold_corr: {threshold_corr}")
                                print(f"attention_head: {attention_head}")
                                print(f"attention_depth: {attention_depth}")


def exp_grid_search_full_bs32(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
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
                            accuracy, loss, drop_column_name = fn_run_experiment(
                                train_data, test_data,
                                drop_column_name,
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
                                print(f"drop_column_name: {drop_column_name}")
                                print(f"emb_dropout: {emb_dropout}")
                                print(f"lr: {lr}")
                                print(f"batch_size: {batch_size}")
                                print(f"threshold_corr: {threshold_corr}")
                                print(f"attention_head: {attention_head}")
                                print(f"attention_depth: {attention_depth}")



def exp_grid_search_full_bs1632_head68_reprod(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
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
                            accuracy, loss, drop_column_name = fn_run_experiment(
                                train_data, test_data,
                                drop_column_name,
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
                                print(f"drop_column_name: {drop_column_name}")
                                print(f"emb_dropout: {emb_dropout}")
                                print(f"lr: {lr}")
                                print(f"batch_size: {batch_size}")
                                print(f"threshold_corr: {threshold_corr}")
                                print(f"attention_head: {attention_head}")
                                print(f"attention_depth: {attention_depth}")





def exp_grid_search_full_bs1632_head68(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
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
                            accuracy, loss, drop_column_name = fn_run_experiment(
                                train_data, test_data,
                                drop_column_name,
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
                                print(f"drop_column_name: {drop_column_name}")
                                print(f"emb_dropout: {emb_dropout}")
                                print(f"lr: {lr}")
                                print(f"batch_size: {batch_size}")
                                print(f"threshold_corr: {threshold_corr}")
                                print(f"attention_head: {attention_head}")
                                print(f"attention_depth: {attention_depth}")


def exp_grid_search_full_bs1632_head68_posembed(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
    fn_run_experiment: Callable,
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
                                accuracy, loss, drop_column_name = fn_run_experiment(
                                    train_data, test_data,
                                    drop_column_name,
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
                                    print(f"drop_column_name: {drop_column_name}")
                                    print(f"emb_dropout: {emb_dropout}")
                                    print(f"lr: {lr}")
                                    print(f"batch_size: {batch_size}")
                                    print(f"threshold_corr: {threshold_corr}")
                                    print(f"attention_head: {attention_head}")
                                    print(f"attention_depth: {attention_depth}")








def exp_grid_search_full_bs32_wd1e_7(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
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
                            accuracy, loss, drop_column_name = fn_run_experiment(
                                train_data, test_data,
                                drop_column_name,
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
                                print(f"drop_column_name: {drop_column_name}")
                                print(f"emb_dropout: {emb_dropout}")
                                print(f"lr: {lr}")
                                print(f"batch_size: {batch_size}")
                                print(f"threshold_corr: {threshold_corr}")
                                print(f"attention_head: {attention_head}")
                                print(f"attention_depth: {attention_depth}")


def exp_grid_search_full_bs32_wd1e_7_head123(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
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
                            accuracy, loss, drop_column_name = fn_run_experiment(
                                train_data, test_data,
                                drop_column_name,
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
                                print(f"drop_column_name: {drop_column_name}")
                                print(f"emb_dropout: {emb_dropout}")
                                print(f"lr: {lr}")
                                print(f"batch_size: {batch_size}")
                                print(f"threshold_corr: {threshold_corr}")
                                print(f"attention_head: {attention_head}")
                                print(f"attention_depth: {attention_depth}")


def exp_grid_search_full_bs32_wd1e_7_head123_optname(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
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
                                accuracy, loss, drop_column_name = fn_run_experiment(
                                    train_data, test_data,
                                    drop_column_name,
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
                                    print(f"drop_column_name: {drop_column_name}")
                                    print(f"emb_dropout: {emb_dropout}")
                                    print(f"lr: {lr}")
                                    print(f"batch_size: {batch_size}")
                                    print(f"optimizer_name: {optimizer_name}")
                                    print(f"threshold_corr: {threshold_corr}")
                                    print(f"attention_head: {attention_head}")
                                    print(f"attention_depth: {attention_depth}")


def exp_grid_search_full_bs1632_head68_posembed_final(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_column_name: List[str],
    fn_run_experiment: Callable,
    note: str = "",
    device: str = "cuda:0",
    amp: bool = False,
):

    list_threshold_corr = [0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.5, 0.0]
    list_lr = [0.00001, 0.00005, 0.0001]
    list_batch_size = [16, 32, 64]
    list_attention_head = [4, 6, 8]
    list_attention_depth = [2, 3]
    list_emb_dropout = [0.5]
    list_weight_decay = [1e-6, 1e-7]
    list_pos_emb_config = [
        {"use_abs_pos_emb": True, "rel_pos_bias": True, "scaled_sinu_pos_emb": False, "post_emb_norm": True},
    ]

    max_epochs = 200
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
                                    accuracy, loss, drop_column_name = fn_run_experiment(
                                        train_data, test_data,
                                        drop_column_name,
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
                                        print(f"drop_column_name: {drop_column_name}")
                                        print(f"emb_dropout: {emb_dropout}")
                                        print(f"lr: {lr}")
                                        print(f"batch_size: {batch_size}")
                                        print(f"threshold_corr: {threshold_corr}")
                                        print(f"attention_head: {attention_head}")
                                        print(f"attention_depth: {attention_depth}")
