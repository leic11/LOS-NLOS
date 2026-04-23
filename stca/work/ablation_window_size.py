# ablation_window_size.py
"""
消融实验 1：时间窗口长度实验（4 特征）
============================

用途：
    探究不同时间窗口长度对 GNSS NLOS 检测模型性能的影响。

实验设置：
    - 使用跨域数据（outdomain 模式）
    - 使用 4 特征输入（4 个原始特征）
    - 窗口长度从 6 到 32，步长为 2（即 6, 8, 10, ..., 32）
    - 固定其他超参数，仅改变窗口长度

输出：
    - 每个窗口长度对应的模型权重和评估结果
    - 性能变化折线图（ACC, PRE, REC, F1）

使用方式：
    python -m work.ablation_window_size
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

# 添加项目路径
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

MODULES_DIR = Path(__file__).parent / "modules"
sys.path.insert(0, str(MODULES_DIR))

STCA_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(STCA_DIR))

from utils.logger_config import setup_logger
from utils.seed_utils import set_seed
from modules.stca_model import STCAModel
from data_loading.main import StaticPreprocessor
from data_loading.constants import DEFAULT_MAX_SATELLITES

logger = setup_logger(__name__)

# ============================================================================
# 实验配置
# ============================================================================

# 窗口长度范围：6 到 32，步长为 2
WINDOW_SIZES = list(range(6, 33, 2))  # 6, 8, 10, ..., 32

# 从 constants.py 导入统一超参数
from modules.constants import LEARNING_RATE, EPOCHS, BATCH_SIZE, RANDOM_SEED, DEVICE

# 固定超参数
CONFIG = {
    "split_mode": "outdomain",  # 跨域数据
    "random_seed": RANDOM_SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "max_satellites": DEFAULT_MAX_SATELLITES,
    # 模型参数（使用 modules/constants.py 默认值）
}

# 输出目录
OUTPUT_DIR = ROOT_DIR / "outputs" / "stca" / "ablation" / "window_size"


def load_or_preprocess_data(window_size, split_mode):
    """加载或预处理数据"""
    npz_filename = f"static_processed_{split_mode}_w{window_size}.npz"
    npz_path = STCA_DIR / npz_filename

    if npz_path.exists():
        logger.info(f"Loading data from {npz_path}")
        data = StaticPreprocessor.load_processed(str(npz_path))
    else:
        logger.info(f"Preprocessing data with window_size={window_size}...")
        data_dir = STCA_DIR.parent / "data for sharing_csv"
        preprocessor = StaticPreprocessor(data_dir=str(data_dir))
        data = preprocessor.process_stca(
            window_size=window_size,
            split_mode=split_mode,
        )
        preprocessor.save_processed(data, str(npz_path))
        logger.info(f"Data saved to {npz_path}")

    return data


def train_and_evaluate(window_size):
    """训练并评估单个窗口长度"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Window Size: {window_size}")
    logger.info(f"{'='*60}")

    # 设置随机种子
    set_seed(CONFIG["random_seed"])

    # 加载数据
    data = load_or_preprocess_data(window_size, CONFIG["split_mode"])

    X_train_spatial = data["X_train_spatial"]
    X_train_temporal = data["X_train_temporal"]
    y_train = data["y_train"]
    X_test_spatial = data["X_test_spatial"]
    X_test_temporal = data["X_test_temporal"]
    y_test = data["y_test"]

    logger.info(f"Data loaded: Train={len(y_train)}, Test={len(y_test)}")

    # 导入默认配置
    from modules.constants import (
        SPATIAL_EMBED_DIM, SPATIAL_NUM_HEADS, SPATIAL_NUM_LAYERS, SPATIAL_DROPOUT,
        TEMPORAL_EMBED_DIM, TEMPORAL_NUM_LAYERS, TEMPORAL_DROPOUT,
        CROSS_ATTN_EMBED_DIM, CROSS_ATTN_NUM_HEADS, CROSS_ATTN_DROPOUT,
        CLASSIFIER_HIDDEN_DIMS, CLASSIFIER_DROPOUT,
    )

    # 构建模型（使用 4 特征输入）
    model = STCAModel(
        input_dim=4,  # 4 特征
        num_classes=2,
        spatial_embed_dim=SPATIAL_EMBED_DIM,
        spatial_num_heads=SPATIAL_NUM_HEADS,
        spatial_num_layers=SPATIAL_NUM_LAYERS,
        spatial_dropout=SPATIAL_DROPOUT,
        temporal_embed_dim=TEMPORAL_EMBED_DIM,
        temporal_num_layers=TEMPORAL_NUM_LAYERS,
        temporal_dropout=TEMPORAL_DROPOUT,
        cross_attn_embed_dim=CROSS_ATTN_EMBED_DIM,
        cross_attn_num_heads=CROSS_ATTN_NUM_HEADS,
        cross_attn_dropout=CROSS_ATTN_DROPOUT,
        classifier_hidden_dims=CLASSIFIER_HIDDEN_DIMS,
        classifier_dropout=CLASSIFIER_DROPOUT,
    )

    logger.info(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    logger.info(f"Using device: {DEVICE}")

    # 训练
    logger.info(f"Training for {CONFIG['epochs']} epochs...")
    history = model.fit(
        X_train_spatial, y_train,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        lr=CONFIG["learning_rate"],
        device=DEVICE,
        verbose=True,  # 显示每个 epoch 的训练进度
        X_train_temporal=X_train_temporal,
    )

    # 评估
    logger.info("Evaluating on test set...")
    metrics = model.evaluate(
        X_test_spatial, y_test,
        device=DEVICE,
        X_test_3d=X_test_temporal,
    )

    # 保存结果
    result = {
        "window_size": window_size,
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "best_train_acc": float(max(history["train_acc"])),
    }

    return result, history


def plot_results(all_results):
    """绘制性能变化折线图"""
    window_sizes = [r["window_size"] for r in all_results]
    accuracies = [r["accuracy"] * 100 for r in all_results]
    precisions = [r["precision"] * 100 for r in all_results]
    recalls = [r["recall"] * 100 for r in all_results]
    f1_scores = [r["f1_score"] * 100 for r in all_results]

    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(12, 8))

    # 绘制 4 条指标曲线
    plt.plot(window_sizes, accuracies, 'o-', linewidth=2.5, markersize=10, label='Accuracy', color='#1f77b4')
    plt.plot(window_sizes, precisions, 's-', linewidth=2.5, markersize=10, label='Precision', color='#ff7f0e')
    plt.plot(window_sizes, recalls, '^-', linewidth=2.5, markersize=10, label='Recall', color='#2ca02c')
    plt.plot(window_sizes, f1_scores, 'd-', linewidth=2.5, markersize=10, label='F1 Score', color='#d62728')

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
    best_idx = np.argmax(f1_scores)
    best_window = window_sizes[best_idx]
    best_f1 = f1_scores[best_idx]

    # 放在图内左上角，固定位置
    plt.figtext(0.01, 0.93,
        f'Best F1: {best_f1:.2f}%\n(Window Size = {best_window})',
        fontsize=12,
        color='red',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        verticalalignment='top'
    )

    plt.xlabel('Window Size', fontsize=14, fontweight='bold')
    plt.ylabel('Performance (%)', fontsize=14, fontweight='bold')
    plt.title('Ablation Study: Effect of Window Size on Model Performance', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')

    # 设置 x 轴刻度为整数
    plt.xticks(window_sizes)

    # 设置 y 轴范围（留出空间给数值标签）
    y_min = min(min(accuracies), min(precisions), min(recalls), min(f1_scores))
    y_max = max(max(accuracies), max(precisions), max(recalls), max(f1_scores))
    y_padding = (y_max - y_min) * 0.15
    plt.ylim(y_min - y_padding * 0.5, min(100, y_max + y_padding + 2))

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "ablation_window_size.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Plot saved to {plot_path}")

    # 保存数据为 JSON
    data = {
        "window_sizes": window_sizes,
        "accuracy": [x / 100 for x in accuracies],  # 恢复为 0-1 范围
        "precision": [x / 100 for x in precisions],
        "recall": [x / 100 for x in recalls],
        "f1_scores": [x / 100 for x in f1_scores],
        "best_window_size": best_window,
        "best_f1_score": best_f1,
    }

    json_path = OUTPUT_DIR / "ablation_window_size_data.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Data saved to {json_path}")


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("Ablation Study 1: Window Size Experiment")
    logger.info("="*60)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for window_size in WINDOW_SIZES:
        result, history = train_and_evaluate(window_size)
        all_results.append(result)

        # 保存单个结果
        result_path = OUTPUT_DIR / f"result_w{window_size}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Result: ACC={result['accuracy']:.4f}, PRE={result['precision']:.4f}, "
                   f"REC={result['recall']:.4f}, F1={result['f1_score']:.4f}")

    # 绘制汇总图
    plot_results(all_results)

    # 输出汇总表格
    logger.info("\n" + "="*60)
    logger.info("Summary Table")
    logger.info("="*60)
    logger.info(f"{'Window Size':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    logger.info("-"*60)
    for r in all_results:
        logger.info(f"{r['window_size']:<15} {r['accuracy']*100:<12.2f} {r['precision']*100:<12.2f} "
                   f"{r['recall']*100:<12.2f} {r['f1_score']*100:<12.2f}")

    logger.info("\nAblation study complete!")


if __name__ == "__main__":
    main()
