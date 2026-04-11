# replot_confusion_matrix.py
"""
混淆矩阵重绘图工具
==================

用途：
    根据 outputs/stca/ablation/modules/ 目录下已保存的 JSON 结果文件，
    重新生成 6 张混淆矩阵图。

使用方式：
    python -m utils.replot_confusion_matrix

输入：
    outputs/stca/ablation/modules/result_{split_mode}_{module_key}.json

输出：
    outputs/stca/ablation/modules/confusion_{split_mode}_{module_key}.png
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
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
RESULTS_DIR = ROOT_DIR / "outputs" / "stca" / "ablation" / "modules"

# 实验配置
MODULE_CONFIGS = {
    "baseline": "Baseline",
    "crossattn": "+CrossAttn",
    "both": "+Both",
}

SPLIT_MODES = ["indomain", "outdomain"]


def load_result(split_mode, module_key):
    """加载 JSON 结果文件"""
    json_path = RESULTS_DIR / f"result_{split_mode}_{module_key}.json"

    if not json_path.exists():
        logger.warning(f"Result file not found: {json_path}")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data


def plot_confusion_matrix_from_json(data, split_mode, module_key):
    """从 JSON 数据绘制混淆矩阵"""
    y_true = np.array(data["y_true"])
    y_pred = np.array(data["y_pred"])
    module_name = data.get("module_name", module_key)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 计算百分比（按行归一化）
    cm_sum = cm.sum(axis=1, keepdims=True)
    # 避免除零
    cm_sum = np.where(cm_sum == 0, 1, cm_sum)
    cm_percent = cm.astype('float') / cm_sum * 100

    plt.figure(figsize=(8, 6))

    # 创建自定义注释：数值 + 百分比
    annot_labels = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_labels[i, j] = f'{cm[i, j]:d}\n{cm_percent[i, j]:.2f}%'

    # 绘制热力图（使用归一化的混淆矩阵用于颜色条）
    ax = sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap='Blues',
                     cbar_kws={'format': '%.0f%%'},
                     xticklabels=['NLOS', 'LOS'], yticklabels=['NLOS', 'LOS'])

    plt.title(f'Confusion Matrix - {split_mode.upper()} - {module_name}',
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # 保存
    plot_path = RESULTS_DIR / f"confusion_{split_mode}_{module_key}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix saved to {plot_path}")
    return cm


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("Replot Confusion Matrices from JSON Results")
    logger.info("="*60)

    if not RESULTS_DIR.exists():
        logger.error(f"Results directory not found: {RESULTS_DIR}")
        return

    # 列出所有结果文件
    result_files = list(RESULTS_DIR.glob("result_*.json"))
    if not result_files:
        logger.error(f"No result files found in {RESULTS_DIR}")
        return

    logger.info(f"Found {len(result_files)} result files")

    success_count = 0
    for split_mode in SPLIT_MODES:
        for module_key in MODULE_CONFIGS.keys():
            data = load_result(split_mode, module_key)

            if data is None:
                logger.warning(f"Skipping {split_mode}_{module_key}: no data")
                continue

            cm = plot_confusion_matrix_from_json(data, split_mode, module_key)

            # 输出简要统计
            total = cm.sum()
            correct = np.trace(cm)
            accuracy = correct / total * 100 if total > 0 else 0
            logger.info(f"  [{split_mode}] {MODULE_CONFIGS[module_key]}: "
                       f"Accuracy = {accuracy:.2f}% ({correct}/{total})")
            success_count += 1

    logger.info("="*60)
    logger.info(f"Replotted {success_count} confusion matrices")
    logger.info("="*60)


if __name__ == "__main__":
    main()
