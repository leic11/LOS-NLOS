# replot_confusion_matrix.py
"""
混淆矩阵重绘图工具
==================

用途：
    1. 模块消融混淆矩阵：从 JSON 结果文件重绘（6 张）
    2. 内外域混淆矩阵：加载模型权重，重新推理生成（2 张）

使用方式：
    # 从项目根目录 DevLab/ 执行
    python -m stca.utils.replot_confusion_matrix

输入：
    outputs/stca/ablation/modules/result_{split_mode}_{module_key}.json
    outputs/stca/final_model_{split_mode}.pth
    stca/static_processed_{split_mode}.npz

输出：
    outputs/stca/ablation/modules/confusion_{split_mode}_{module_key}.png
    outputs/stca/figures/confusion_matrix_{split_mode}.png
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 添加项目路径
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

STCA_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(STCA_DIR))

from utils.logger_config import setup_logger

logger = setup_logger(__name__)

# 输入输出目录
RESULTS_DIR = ROOT_DIR / "outputs" / "stca" / "ablation" / "baseline"
FIGURES_DIR = ROOT_DIR / "outputs" / "stca" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 实验配置
MODULE_CONFIGS = {
    "concat": "Concat",
    "crossattn": "+CrossAttn",
    "full_stca": "Full STCA",
}

SPLIT_MODES = ["indomain", "outdomain"]
DEVICE = torch.device("cpu")

# 矩阵内文字统一字号
ANNOT_FONTSIZE = 14
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 13


def _draw_cm(cm, title, save_path):
    """绘制单张混淆矩阵并保存"""
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_sum = np.where(cm_sum == 0, 1, cm_sum)
    cm_percent = cm.astype('float') / cm_sum * 100

    plt.figure(figsize=(7, 5.5))

    annot_labels = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_labels[i, j] = f'{cm[i, j]:d}\n{cm_percent[i, j]:.2f}%'

    sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap='Blues',
                cbar_kws={'format': '%.0f%%'},
                xticklabels=['NLOS', 'LOS'], yticklabels=['NLOS', 'LOS'],
                annot_kws={'size': ANNOT_FONTSIZE})

    plt.ylabel('True Label', fontsize=LABEL_FONTSIZE)
    plt.xlabel('Predicted Label', fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


# ========== 1. 模块消融混淆矩阵（从 JSON） ==========

def load_result(module_key):
    """加载 JSON 结果文件"""
    json_path = RESULTS_DIR / f"result_{module_key}.json"

    if not json_path.exists():
        logger.warning(f"Result file not found: {json_path}")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data


def plot_module_ablation_cm():
    """绘制模块消融混淆矩阵"""
    logger.info("--- Module Ablation Confusion Matrices ---")

    if not RESULTS_DIR.exists():
        logger.warning(f"Results directory not found: {RESULTS_DIR}")
        return

    success_count = 0

    for module_key, module_name in MODULE_CONFIGS.items():
        data = load_result(module_key)
        if data is None:
            continue

        y_true = np.array(data["y_true"])
        y_pred = np.array(data["y_pred"])
        cm = confusion_matrix(y_true, y_pred)

        plot_path = FIGURES_DIR / f"confusion_{module_key}.png"
        _draw_cm(cm, module_name, plot_path)

        total = cm.sum()
        correct = np.trace(cm)
        accuracy = correct / total * 100 if total > 0 else 0
        logger.info(f"  {module_name}: Accuracy = {accuracy:.2f}% ({correct}/{total})")
        success_count += 1

    logger.info(f"Replotted {success_count} module ablation confusion matrices")


# ========== 2. 内外域混淆矩阵（从模型推理） ==========

from modules.stca_model import STCAModel
from data_loading.main import StaticPreprocessor
from data_loading.constants import INPUT_DIM, NUM_CLASSES


def load_model_and_data(split_mode):
    """加载模型权重和测试数据"""
    npz_path = STCA_DIR / f"static_processed_{split_mode}.npz"
    if not npz_path.exists():
        logger.error(f"Data file not found: {npz_path}")
        return None, None, None

    data = StaticPreprocessor.load_processed(str(npz_path))
    X_test_spatial = data["X_test_spatial"]
    X_test_temporal = data["X_test_temporal"]
    y_test = data["y_test"]

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
    """运行推理，返回预测标签"""
    all_probs = []
    batch_size = 128

    with torch.no_grad():
        for i in range(0, len(X_spatial), batch_size):
            x_s = torch.FloatTensor(X_spatial[i:i+batch_size]).to(DEVICE)
            x_t = torch.FloatTensor(X_temporal[i:i+batch_size]).to(DEVICE)
            probs = model(x_spatial=x_s, x_temporal=x_t).squeeze(-1)
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_probs) >= 0.5).astype(int)


def plot_split_cm():
    """绘制内域/外域混淆矩阵"""
    logger.info("--- Indomain / Outdomain Confusion Matrices ---")

    for split_mode in SPLIT_MODES:
        model, (X_s, X_t), y_test = load_model_and_data(split_mode)
        if model is None:
            continue

        y_pred = predict(model, X_s, X_t)
        cm = confusion_matrix(y_test, y_pred)

        out_path = FIGURES_DIR / f"confusion_matrix_{split_mode}.png"
        _draw_cm(cm, split_mode, out_path)

        total = cm.sum()
        correct = np.trace(cm)
        accuracy = correct / total * 100 if total > 0 else 0
        logger.info(f"  [{split_mode}] Accuracy = {accuracy:.2f}%")


# ========== 主入口 ==========

def main():
    logger.info("=" * 60)
    logger.info("Confusion Matrix Replot (All)")
    logger.info("=" * 60)

    plot_module_ablation_cm()
    plot_split_cm()

    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
