# replot_window_ablation.py
"""
时间窗口消融实验重绘图工具
==========================

用途：
    根据 outputs/stca/ablation/window_size/ 目录下已保存的 JSON 数据文件，
    重新生成时间窗口长度消融实验的折线图。

使用方式：
    python -m utils.replot_window_ablation

输入：
    outputs/stca/ablation/window_size/ablation_window_size_data.json

输出：
    outputs/stca/ablation/window_size/ablation_window_size.png
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加项目路径
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.logger_config import setup_logger

logger = setup_logger(__name__)

# 输入输出目录
RESULTS_DIR = ROOT_DIR / "outputs" / "stca" / "ablation" / "window_size"

# 绘图配置
PLOT_CONFIG = {
    "figsize": (12, 8),
    "dpi": 150,
    "font_size": 14,
    "line_width": 2.5,
    "marker_size": 10,
    "text_offset": 0.8,  # 数值标签偏移量
}


def load_data():
    """加载 JSON 数据文件"""
    json_path = RESULTS_DIR / "ablation_window_size_data.json"

    if not json_path.exists():
        logger.warning(f"Data file not found: {json_path}")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data


def plot_window_ablation(data):
    """
    绘制时间窗口消融实验折线图

    Args:
        data: 包含 window_sizes, accuracy, precision, recall, f1_scores 的字典
    """
    window_sizes = data["window_sizes"]
    accuracies = data["accuracy"]
    precisions = data["precision"]
    recalls = data["recall"]
    f1_scores = data["f1_scores"]

    # 找到最佳 F1 分数点
    best_idx = np.argmax(f1_scores)
    best_window = window_sizes[best_idx]
    best_f1 = f1_scores[best_idx]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=PLOT_CONFIG["figsize"])

    # 绘制 4 条指标曲线
    acc_line = plt.plot(window_sizes, accuracies, 'o-', linewidth=PLOT_CONFIG["line_width"],
                        markersize=PLOT_CONFIG["marker_size"], label='Accuracy', color='#1f77b4')
    pre_line = plt.plot(window_sizes, precisions, 's-', linewidth=PLOT_CONFIG["line_width"],
                        markersize=PLOT_CONFIG["marker_size"], label='Precision', color='#ff7f0e')
    rec_line = plt.plot(window_sizes, recalls, '^-', linewidth=PLOT_CONFIG["line_width"],
                        markersize=PLOT_CONFIG["marker_size"], label='Recall', color='#2ca02c')
    f1_line = plt.plot(window_sizes, f1_scores, 'd-', linewidth=PLOT_CONFIG["line_width"],
                       markersize=PLOT_CONFIG["marker_size"], label='F1 Score', color='#d62728')

    # 在每个数据点上方添加具体数值标签（黑色字体 + 与线条同色的背景）
    text_offset = 2.0  # 增大偏移量

    for i, (w, acc) in enumerate(zip(window_sizes, accuracies)):
        plt.annotate(f'{acc:.2f}', xy=(w, acc), xytext=(0, text_offset),
                     textcoords='offset points', ha='center', va='bottom',
                     fontsize=11, color='black', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#1f77b4', alpha=0.8, edgecolor='none'))

    for i, (w, pre) in enumerate(zip(window_sizes, precisions)):
        plt.annotate(f'{pre:.2f}', xy=(w, pre), xytext=(0, text_offset),
                     textcoords='offset points', ha='center', va='bottom',
                     fontsize=11, color='black', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#ff7f0e', alpha=0.8, edgecolor='none'))

    for i, (w, rec) in enumerate(zip(window_sizes, recalls)):
        plt.annotate(f'{rec:.2f}', xy=(w, rec), xytext=(0, text_offset),
                     textcoords='offset points', ha='center', va='bottom',
                     fontsize=11, color='black', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#2ca02c', alpha=0.8, edgecolor='none'))

    for i, (w, f1) in enumerate(zip(window_sizes, f1_scores)):
        plt.annotate(f'{f1:.2f}', xy=(w, f1), xytext=(0, text_offset),
                     textcoords='offset points', ha='center', va='bottom',
                     fontsize=11, color='black', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#d62728', alpha=0.8, edgecolor='none'))

    # 标注最佳 F1 分数点（图像内左上方显示，备注窗口大小）
    # 放在图内左上角，固定位置
    plt.figtext(0.06, 0.93,
        f'Best F1: {best_f1:.2f}%\n(Window Size = {best_window})',
        fontsize=12,
        color='red',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        verticalalignment='top'
    )

    # 设置标签和标题
    plt.xlabel('Window Size', fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
    plt.ylabel('Performance (%)', fontsize=PLOT_CONFIG["font_size"], fontweight='bold')
    plt.title('Ablation Study: Effect of Window Size on Model Performance',
              fontsize=16, fontweight='bold')

    # 图例
    plt.legend(fontsize=11, loc='lower right')

    # 设置 x 轴刻度为整数
    plt.xticks(window_sizes)

    # 设置 y 轴范围（留出空间给数值标签）
    y_min = min(min(accuracies), min(precisions), min(recalls), min(f1_scores))
    y_max = max(max(accuracies), max(precisions), max(recalls), max(f1_scores))
    y_padding = (y_max - y_min) * 0.15
    plt.ylim(y_min - y_padding * 0.5, min(100, y_max + y_padding + 2))

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # 保存
    plot_path = RESULTS_DIR / "ablation_window_size.png"
    plt.savefig(plot_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    plt.close()

    logger.info(f"Plot saved to {plot_path}")

    # 同时保存带有数值标签的数据
    labeled_data = {
        "window_sizes": window_sizes,
        "accuracy": accuracies,
        "precision": precisions,
        "recall": recalls,
        "f1_scores": f1_scores,
        "best_window_size": best_window,
        "best_f1_score": best_f1,
    }

    # 更新 JSON 文件（包含最佳结果）
    json_path = RESULTS_DIR / "ablation_window_size_data.json"
    with open(json_path, 'w') as f:
        json.dump(labeled_data, f, indent=2)

    return plot_path


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("Replot Window Size Ablation Experiment")
    logger.info("="*60)

    data = load_data()

    if data is None:
        logger.error("No data found. Please run the window size ablation experiment first.")
        return

    # 输出数据表格
    logger.info("\nWindow Size Ablation Results:")
    logger.info(f"{'Window Size':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    logger.info("-"*60)
    for i, w in enumerate(data["window_sizes"]):
        logger.info(f"{w:<15} {data['accuracy'][i]*100:<12.2f} {data['precision'][i]*100:<12.2f} "
                   f"{data['recall'][i]*100:<12.2f} {data['f1_scores'][i]*100:<12.2f}")

    logger.info("-"*60)
    logger.info(f"Best Window Size: {data['best_window_size']} (F1 Score: {data['best_f1_score']*100:.2f}%)")

    # 绘制折线图
    plot_path = plot_window_ablation(data)

    logger.info("="*60)
    logger.info(f"Replotted to: {plot_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
