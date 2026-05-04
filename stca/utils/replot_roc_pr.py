# replot_roc_pr.py
"""
ROC 曲线与 PR 曲线重绘工具
==========================

用途：
    加载已保存的 indomain / outdomain 模型权重和预处理数据，
    重新运行推理获取概率输出，绘制并排对比的 ROC 曲线和 PR 曲线，
    用于论文图 4-2（ROC 曲线对比）和图 4-3（PR 曲线对比）。

使用方式：
    # 从项目根目录 DevLab/ 执行
    python -m stca.utils.replot_roc_pr

输入：
    outputs/stca/final_model_indomain.pth
    outputs/stca/final_model_outdomain.pth
    stca/static_processed_indomain.npz
    stca/static_processed_outdomain.npz

输出：
    outputs/stca/figures/roc_curve_comparison.png
    outputs/stca/figures/pr_curve_comparison.png
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 路径设置
ROOT_DIR = Path(__file__).parent.parent.parent
STCA_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(STCA_DIR))

from utils.logger_config import setup_logger
from modules.stca_model import STCAModel
from data_loading.main import StaticPreprocessor
from data_loading.constants import INPUT_DIM, NUM_CLASSES

logger = setup_logger(__name__)

# 输出目录
OUTPUT_DIR = ROOT_DIR / "outputs" / "stca" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_MODES = ["indomain", "outdomain"]
DEVICE = torch.device("cpu")


def load_model_and_data(split_mode):
    """加载模型权重和测试数据"""
    # 加载预处理数据
    npz_path = STCA_DIR / f"static_processed_{split_mode}.npz"
    if not npz_path.exists():
        logger.error(f"Data file not found: {npz_path}")
        return None, None, None

    data = StaticPreprocessor.load_processed(str(npz_path))
    X_test_spatial = data["X_test_spatial"]
    X_test_temporal = data["X_test_temporal"]
    y_test = data["y_test"]

    # 构建模型并加载权重
    model = STCAModel(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    ckpt_path = ROOT_DIR / "outputs" / "stca" / f"final_model_{split_mode}.pth"
    if not ckpt_path.exists():
        logger.error(f"Model checkpoint not found: {ckpt_path}")
        return None, None, None

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    logger.info(f"[{split_mode}] Model loaded, test samples: {len(y_test)}")
    return model, (X_test_spatial, X_test_temporal), y_test


def predict(model, X_spatial, X_temporal):
    """运行推理，返回概率数组"""
    all_probs = []
    batch_size = 128

    with torch.no_grad():
        for i in range(0, len(X_spatial), batch_size):
            x_s = torch.FloatTensor(X_spatial[i:i+batch_size]).to(DEVICE)
            x_t = torch.FloatTensor(X_temporal[i:i+batch_size]).to(DEVICE)
            probs = model(x_spatial=x_s, x_temporal=x_t).squeeze(-1)
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)


def plot_all_curves(results):
    """分别绘制 4 张独立图片：ROC×2 + PR×2"""
    labels = {'indomain': 'Indomain', 'outdomain': 'Outdomain'}

    for split_mode, res in results.items():
        y_true, y_proba = res["y_true"], res["y_proba"]
        tag = labels[split_mode]

        # --- ROC ---
        fig, ax = plt.subplots(figsize=(7, 5.5))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2.5,
                label=f'ROC curve (AUC = {roc_auc_val:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"roc_curve_{split_mode}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curve saved to {out_path}")

        # --- PR ---
        fig, ax = plt.subplots(figsize=(7, 5.5))
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
        pr_auc_val = average_precision_score(y_true, y_proba)
        ax.plot(recall_curve, precision_curve, color='green', lw=2.5,
                label=f'PR curve (AUC = {pr_auc_val:.4f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.legend(loc='lower left', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"pr_curve_{split_mode}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"PR curve saved to {out_path}")


def main():
    logger.info("=" * 60)
    logger.info("ROC & PR Curve Comparison Replot")
    logger.info("=" * 60)

    results = {}
    for split_mode in SPLIT_MODES:
        model, (X_s, X_t), y_test = load_model_and_data(split_mode)
        if model is None:
            logger.error(f"Skipping {split_mode}: failed to load model/data")
            continue

        y_proba = predict(model, X_s, X_t)
        results[split_mode] = {"y_true": y_test, "y_proba": y_proba}

        from sklearn.metrics import accuracy_score, f1_score
        y_pred = (y_proba >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        logger.info(f"[{split_mode}] Acc={acc:.4f}, F1={f1:.4f}")

    if len(results) < 2:
        logger.error("Need both indomain and outdomain results to plot comparison")
        return

    plot_all_curves(results)

    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
