from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import GNSSCombinedDataset, combined_collate_fn
from .engine import evaluate, train_one_epoch
from .model import LOSNLOSModel
from .plotting import save_curve


def run_training(config: TrainConfig) -> None:
    config.outdir.mkdir(parents=True, exist_ok=True)
    config.figdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files = config.resolve_train_files()
    test_files = config.resolve_test_files()

    print(f"[RUN] EXP_NAME={config.exp_name}")
    print(f"[RUN] OUTDIR={config.outdir}")
    print(f"[RUN] FIGDIR={config.figdir}")
    print(f"[RUN] device={device}")
    print(f"[RUN] train_files={train_files}")
    print(f"[RUN] test_files={test_files}")
    print("-" * 80)

    ds_train = GNSSCombinedDataset(
        train_files,
        history_len=config.history_len,
        feature_cols=config.feature_cols,
        label_col=config.label_col,
        split_by_point=config.split_by_point,
    )
    ds_test = GNSSCombinedDataset(
        test_files,
        history_len=config.history_len,
        feature_cols=config.feature_cols,
        label_col=config.label_col,
        split_by_point=config.split_by_point,
        mean=ds_train.mean,
        std=ds_train.std,
    )

    count_label1_train = sum(1 for s in ds_train if s["label"] == 1)
    count_label0_train = sum(1 for s in ds_train if s["label"] == 0)
    count_label1_test = sum(1 for s in ds_test if s["label"] == 1)
    count_label0_test = sum(1 for s in ds_test if s["label"] == 0)

    print(f"Train Dataset: label=1 -> {count_label1_train}, label=0 -> {count_label0_train}")
    print(f"Test Dataset:  label=1 -> {count_label1_test}, label=0 -> {count_label0_test}")

    loader_tr = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True, collate_fn=combined_collate_fn)
    loader_te = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False, collate_fn=combined_collate_fn)

    total = count_label1_train + count_label0_train
    pos_weight = count_label0_train / max(1, total)
    neg_weight = count_label1_train / max(1, total)
    weights = torch.tensor([pos_weight, neg_weight], dtype=torch.float32).to(device)
    print(f"Loss weights: {weights}")

    model = LOSNLOSModel(
        lstm_hidden=config.lstm_hidden,
        lstm_layers=config.lstm_layers,
        attn_hidden=config.attn_hidden,
        attn_heads=config.attn_heads,
        ff_hidden=config.ff_hidden,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    es, tr_ls, vl_ls, tr_as, vl_as, tr_f1, vl_f1 = [], [], [], [], [], [], []

    for ep in range(1, config.epochs + 1):
        tl, ta, _, _, t_f1 = train_one_epoch(
            model, loader_tr, optimizer, criterion, device, lambda_reg=config.train_lambda_reg
        )
        vl, va, _, _, v_f1 = evaluate(
            model, loader_te, criterion, device, lambda_reg=config.eval_lambda_reg
        )
        scheduler.step()

        print(
            f"Ep{ep:02d} | Train L={tl:.4f} Acc={ta:.4f} F1={t_f1:.4f} | "
            f"Val L={vl:.4f} Acc={va:.4f} F1={v_f1:.4f}"
        )

        es.append(ep)
        tr_ls.append(tl)
        vl_ls.append(vl)
        tr_as.append(ta)
        vl_as.append(va)
        tr_f1.append(t_f1)
        vl_f1.append(v_f1)

    log_df = pd.DataFrame(
        {
            "epoch": es,
            "train_loss": tr_ls,
            "val_loss": vl_ls,
            "train_acc": tr_as,
            "val_acc": vl_as,
            "train_f1": tr_f1,
            "val_f1": vl_f1,
        }
    )
    log_df.to_csv(config.outdir / "training_log.csv", index=False)
    print(f"[SAVE] training_log.csv -> {config.outdir / 'training_log.csv'}")

    save_curve(es, tr_ls, vl_ls, "Loss", config.figdir / "loss_curve.png", "Train Loss", "Val Loss")
    print(f"[SAVE] {config.figdir / 'loss_curve.png'}")
    save_curve(es, tr_f1, vl_f1, "F1 Score", config.figdir / "f1_curve.png", "Train F1", "Val F1")
    print(f"[SAVE] {config.figdir / 'f1_curve.png'}")
    save_curve(es, tr_as, vl_as, "Accuracy", config.figdir / "acc_curve.png", "Train Acc", "Val Acc")
    print(f"[SAVE] {config.figdir / 'acc_curve.png'}")

    model.eval()
    with torch.no_grad():
        for batch in loader_te:
            time_data = batch["time_history"].to(device)
            space_data = batch["space_global"].to(device)
            space_mask = (space_data.abs().sum(-1) == 0).to(device)
            logits, _, fused = model(time_data, space_data, space_mask, return_fused=True)
            sparse_z = model.regularizer.prox(fused)
            print("Sparse feature vector preview:", sparse_z[:4, :8])
            break

    ckpt_path = config.outdir / "final_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    print(f"[SAVE] {ckpt_path}")
    print("-" * 80)
    print("[RUN] Finished.")
