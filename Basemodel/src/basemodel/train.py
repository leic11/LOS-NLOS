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

    print(f"[RUN] EXP_NAME={config.exp_name}")
    print(f"[RUN] OUTDIR={config.outdir}")
    print(f"[RUN] FIGDIR={config.figdir}")
    print(f"[RUN] device={device}")
    print(f"[RUN] data_dir={config.data_dir}")
    print(f"[RUN] split_mode={config.split_mode}")
    print(f"[RUN] train_locations={config.get_train_locations()}")
    print(f"[RUN] test_locations={config.get_test_locations()}")
    print("-" * 80)

    # 加载训练集
    ds_train = GNSSCombinedDataset(
        data_dir=str(config.data_dir),
        history_len=config.history_len,
        feature_cols=config.feature_cols,
        label_col=config.label_col,
        split_mode=config.split_mode,
        train_locations=config.get_train_locations(),
        test_locations=None,
        split_by_point=config.split_by_point,
    )

    # 加载测试集
    ds_test = GNSSCombinedDataset(
        data_dir=str(config.data_dir),
        history_len=config.history_len,
        feature_cols=config.feature_cols,
        label_col=config.label_col,
        split_mode=config.split_mode,
        train_locations=None,
        test_locations=config.get_test_locations(),
        split_by_point=config.split_by_point,
        mean=ds_train.mean,
        std=ds_train.std,
    )

    # 加载验证集（仅 outdomain 模式）
    if config.split_mode == "outdomain" and config.get_val_locations():
        ds_val = GNSSCombinedDataset(
            data_dir=str(config.data_dir),
            history_len=config.history_len,
            feature_cols=config.feature_cols,
            label_col=config.label_col,
            split_mode=config.split_mode,
            train_locations=None,
            test_locations=config.get_val_locations(),
            split_by_point=config.split_by_point,
            mean=ds_train.mean,
            std=ds_train.std,
        )
        use_val = True
        print(f"[INFO] Validation locations={config.get_val_locations()}")
    else:
        ds_val = None
        use_val = False
        print("[INFO] No separate validation set (using test set for monitoring)")

    count_label1_train = sum(1 for s in ds_train if s["label"] == 1)
    count_label0_train = sum(1 for s in ds_train if s["label"] == 0)
    count_label1_test = sum(1 for s in ds_test if s["label"] == 1)
    count_label0_test = sum(1 for s in ds_test if s["label"] == 0)

    print(f"Train Dataset: label=1 -> {count_label1_train}, label=0 -> {count_label0_train}")
    print(f"Test Dataset:  label=1 -> {count_label1_test}, label=0 -> {count_label0_test}")

    loader_tr = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True, collate_fn=combined_collate_fn)
    loader_te = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False, collate_fn=combined_collate_fn)
    if use_val:
        loader_val = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False, collate_fn=combined_collate_fn)
    else:
        loader_val = loader_te  # 使用测试集作为验证集

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
    best_val_acc = 0
    best_model_state = None

    for ep in range(1, config.epochs + 1):
        tl, ta, _, _, t_f1 = train_one_epoch(
            model, loader_tr, optimizer, criterion, device, lambda_reg=config.train_lambda_reg
        )

        # 在验证集上评估（如果有独立验证集）或在测试集上评估
        vl, va, _, _, v_f1 = evaluate(
            model, loader_val, criterion, device, lambda_reg=config.eval_lambda_reg
        )
        scheduler.step()

        # 保存验证集上最好的模型
        if va > best_val_acc:
            best_val_acc = va
            best_model_state = model.state_dict().copy()

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

    # 保存训练日志
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

    # 保存绘图
    save_curve(es, tr_ls, vl_ls, "Loss", config.figdir / "loss_curve.png", "Train Loss", "Val Loss")
    print(f"[SAVE] {config.figdir / 'loss_curve.png'}")
    save_curve(es, tr_f1, vl_f1, "F1 Score", config.figdir / "f1_curve.png", "Train F1", "Val F1")
    print(f"[SAVE] {config.figdir / 'f1_curve.png'}")
    save_curve(es, tr_as, vl_as, "Accuracy", config.figdir / "acc_curve.png", "Train Acc", "Val Acc")
    print(f"[SAVE] {config.figdir / 'acc_curve.png'}")

    # 保存最好的模型
    if best_model_state is not None:
        ckpt_path = config.outdir / "best_model.pt"
        torch.save({"model_state_dict": best_model_state}, ckpt_path)
        print(f"[SAVE] Best model: {ckpt_path}")

    # 也保存最终模型
    model.eval()
    final_ckpt_path = config.outdir / "final_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, final_ckpt_path)
    print(f"[SAVE] {final_ckpt_path}")

    # 在独立测试集上最终评估
    if use_val:
        print("-" * 80)
        print("[INFO] Evaluating on test set...")
        test_loss, test_acc, _, _, test_f1 = evaluate(
            model, loader_te, criterion, device, lambda_reg=config.eval_lambda_reg
        )
        print(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}, Test F1={test_f1:.4f}")

    print("-" * 80)
    print("[RUN] Finished.")
