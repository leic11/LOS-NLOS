# ablation_baseline.py
"""
基线模型对比实验（3 种模型变体）
============================

用途：
    对比不同模型架构对 GNSS NLOS 检测模型性能的影响。

实验设置：
    - 使用 outdomain 数据划分模式
    - 固定超参数（来自 modules/constants.py）
    - 3 种模型变体对比

3 种模型架构：
    1. Concat：空间 + 时间 → 拼接 → 分类器
    2. CrossAttn：空间 + 时间 → 交叉注意力 → 分类器
    3. Full STCA：空间 + 时间 → 交叉注意力 → 稀疏表示 → 分类器

输出：
    - 汇总表格（模型 | Acc | Pre | Rec | F1）
    - 混淆矩阵（5 个子图）

使用方式：
    python -m work.ablation_baseline
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
from data_loading.constants import DEFAULT_MAX_SATELLITES, DEFAULT_WINDOW_SIZE

logger = setup_logger(__name__)

# ============================================================================
# 基准配置（来自 modules/constants.py）
# ============================================================================

from modules.constants import (
    SPATIAL_EMBED_DIM, SPATIAL_NUM_HEADS, SPATIAL_NUM_LAYERS, SPATIAL_D_FF, SPATIAL_DROPOUT,
    TEMPORAL_EMBED_DIM, TEMPORAL_NUM_LAYERS, TEMPORAL_DROPOUT,
    CROSS_ATTN_EMBED_DIM, CROSS_ATTN_NUM_HEADS, CROSS_ATTN_DROPOUT,
    SPARSE_EMBED_DIM,
    CLASSIFIER_HIDDEN_DIMS, CLASSIFIER_DROPOUT,
    LEARNING_RATE, EPOCHS, BATCH_SIZE, RANDOM_SEED, DEVICE,
)

BASE_CONFIG = {
    "spatial_embed_dim": SPATIAL_EMBED_DIM,       
    "spatial_num_heads": SPATIAL_NUM_HEADS,       
    "spatial_num_layers": SPATIAL_NUM_LAYERS,     
    "spatial_d_ff": SPATIAL_D_FF,                 
    "spatial_dropout": SPATIAL_DROPOUT,           
    "temporal_embed_dim": TEMPORAL_EMBED_DIM,     
    "temporal_num_layers": TEMPORAL_NUM_LAYERS,   
    "temporal_dropout": TEMPORAL_DROPOUT,         
    "cross_attn_embed_dim": CROSS_ATTN_EMBED_DIM, 
    "cross_attn_num_heads": CROSS_ATTN_NUM_HEADS, 
    "cross_attn_dropout": CROSS_ATTN_DROPOUT,     
    "sparse_embed_dim": SPARSE_EMBED_DIM,         
    "classifier_hidden_dims": CLASSIFIER_HIDDEN_DIMS,  
    "classifier_dropout": CLASSIFIER_DROPOUT,     
    "learning_rate": LEARNING_RATE,               
    "batch_size": BATCH_SIZE,                     
    "epochs": EPOCHS,                             
    "random_seed": RANDOM_SEED,                  
    "max_satellites": DEFAULT_MAX_SATELLITES,
    "window_size": DEFAULT_WINDOW_SIZE,
    "split_mode": "outdomain",
    "input_dim": 4,  
    "num_classes": 2,
}

# ============================================================================
# 基线模型配置
# ============================================================================

BASELINE_CONFIGS = {
    "concat": {
        "name": "Concat",
        "description": "空间 + 时间拼接",
        "use_cross_attention": False,
        "use_sparse_representation": False,
    },
    "crossattn": {
        "name": "CrossAttn",
        "description": "交叉注意力融合",
        "use_cross_attention": True,
        "use_sparse_representation": False,
    },
    "full_stca": {
        "name": "Full STCA",
        "description": "完整模型（交叉 + 稀疏）",
        "use_cross_attention": True,
        "use_sparse_representation": True,
    },
}

# 输出目录
OUTPUT_DIR = ROOT_DIR / "outputs" / "stca" / "ablation" / "baseline"


def load_or_preprocess_data(split_mode, window_size):
    """加载或预处理数据"""
    npz_filename = f"static_processed_{split_mode}.npz"
    npz_path = STCA_DIR / npz_filename

    if npz_path.exists():
        logger.info(f"Loading data from {npz_path}")
        data = StaticPreprocessor.load_processed(str(npz_path))
    else:
        logger.info(f"Preprocessing data with split_mode={split_mode}...")
        data_dir = STCA_DIR.parent / "data for sharing_csv"
        preprocessor = StaticPreprocessor(data_dir=str(data_dir))
        data = preprocessor.process_stca(
            window_size=window_size,
            split_mode=split_mode,
        )
        preprocessor.save_processed(data, str(npz_path))
        logger.info(f"Data saved to {npz_path}")

    return data


def create_model(model_key, model_config):
    """根据配置创建模型"""
    model = STCAModel(
        input_dim=BASE_CONFIG["input_dim"],
        num_classes=BASE_CONFIG["num_classes"],
        spatial_embed_dim=BASE_CONFIG["spatial_embed_dim"],
        spatial_num_heads=BASE_CONFIG["spatial_num_heads"],
        spatial_num_layers=BASE_CONFIG["spatial_num_layers"],
        spatial_d_ff=BASE_CONFIG["spatial_d_ff"],
        spatial_dropout=BASE_CONFIG["spatial_dropout"],
        temporal_embed_dim=BASE_CONFIG["temporal_embed_dim"],
        temporal_num_layers=BASE_CONFIG["temporal_num_layers"],
        temporal_dropout=BASE_CONFIG["temporal_dropout"],
        cross_attn_embed_dim=BASE_CONFIG["cross_attn_embed_dim"],
        cross_attn_num_heads=BASE_CONFIG["cross_attn_num_heads"],
        cross_attn_dropout=BASE_CONFIG["cross_attn_dropout"],
        classifier_hidden_dims=BASE_CONFIG["classifier_hidden_dims"],
        classifier_dropout=BASE_CONFIG["classifier_dropout"],
        use_cross_attention=model_config["use_cross_attention"],
        use_sparse_representation=model_config["use_sparse_representation"],
    )

    return model


def train_and_evaluate(model_key, model_config):
    """训练并评估单个模型配置"""
    logger.info(f"\n{'='*60}")
    logger.info(f"模型：{model_config['name']} - {model_config['description']}")
    logger.info(f"{'='*60}")

    # 设置随机种子
    set_seed(BASE_CONFIG["random_seed"])

    # 加载数据
    data = load_or_preprocess_data(BASE_CONFIG["split_mode"], BASE_CONFIG["window_size"])

    X_train_spatial = data["X_train_spatial"]
    X_train_temporal = data["X_train_temporal"]
    y_train = data["y_train"]
    X_test_spatial = data["X_test_spatial"]
    X_test_temporal = data["X_test_temporal"]
    y_test = data["y_test"]

    logger.info(f"Data: Train={len(y_train)}, Test={len(y_test)}")

    # 创建模型
    model = create_model(model_key, model_config)

    logger.info(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    logger.info(f"Using device: {DEVICE}")

    # 训练
    logger.info(f"Training for {BASE_CONFIG['epochs']} epochs...")
    history = model.fit(
        X_train_spatial, y_train,
        epochs=BASE_CONFIG["epochs"],
        batch_size=BASE_CONFIG["batch_size"],
        lr=BASE_CONFIG["learning_rate"],
        device=DEVICE,
        verbose=False,
        X_train_temporal=X_train_temporal,
    )

    # 评估
    logger.info("Evaluating on test set...")
    metrics = model.evaluate(
        X_test_spatial, y_test,
        device=DEVICE,
        X_test_3d=X_test_temporal,
    )

    # 获取预测结果用于绘制混淆矩阵
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(0, len(X_test_spatial), 64):
            end_idx = min(i + 64, len(X_test_spatial))
            x_spatial_batch = torch.FloatTensor(X_test_spatial[i:end_idx]).to(DEVICE)
            x_temporal_batch = torch.FloatTensor(X_test_temporal[i:end_idx]).to(DEVICE)

            outputs = model(x_spatial=x_spatial_batch, x_temporal=x_temporal_batch)

            probs = outputs.squeeze(-1)
            preds = (probs >= 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_test[i:end_idx])

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)

    result = {
        "model_key": model_key,
        "model_name": model_config["name"],
        "description": model_config["description"],
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "y_pred": y_pred.tolist(),
        "y_true": y_true.tolist(),
    }

    logger.info(f"Result: ACC={result['accuracy']:.4f}, PRE={result['precision']:.4f}, "
               f"REC={result['recall']:.4f}, F1={result['f1_score']:.4f}")

    return result


def plot_confusion_matrix(y_true, y_pred, model_key, model_name):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    # 计算百分比（按行归一化）
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(6, 5))

    # 创建自定义注释：数值 + 百分比
    annot_labels = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_labels[i, j] = f'{cm[i, j]:d}\n{cm_percent[i, j]:.1f}%'

    # 绘制热力图
    sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap='Blues',
                cbar_kws={'format': '%.0f%%'},
                xticklabels=['NLOS', 'LOS'], yticklabels=['NLOS', 'LOS'])

    plt.title(f'{model_name}', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=10)
    plt.xlabel('Predicted Label', fontsize=10)

    plot_path = OUTPUT_DIR / f"confusion_{model_key}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix saved to {plot_path}")
    return cm


def plot_summary_comparison(all_results):
    """绘制性能对比柱状图（带绝对提升增量标注，两张独立图片）"""
    model_names = [r["model_name"] for r in all_results]
    accuracies = [r["accuracy"] for r in all_results]
    f1_scores = [r["f1_score"] for r in all_results]

    baseline_acc = accuracies[0]
    baseline_f1 = f1_scores[0]
    acc_gains = [a - baseline_acc for a in accuracies]
    f1_gains = [f - baseline_f1 for f in f1_scores]

    # === 图1：Accuracy ===
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, acc, gain) in enumerate(zip(model_names, accuracies, acc_gains)):
        color = '#1f77b4' if i == 0 else '#2ca02c'
        ax.bar(name, acc * 100, color=color, width=0.45)
        label = f'{acc * 100:.2f}%' if i == 0 else f'{acc * 100:.2f}%\n(+{gain * 100:.2f} pp)'
        ax.text(i, acc * 100 + 0.8, label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "baseline_accuracy.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Accuracy plot saved to {plot_path}")

    # === 图2：F1 Score ===
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, f1, gain) in enumerate(zip(model_names, f1_scores, f1_gains)):
        color = '#ff7f0e' if i == 0 else '#2ca02c'
        ax.bar(name, f1 * 100, color=color, width=0.45)
        label = f'{f1 * 100:.2f}%' if i == 0 else f'{f1 * 100:.2f}%\n(+{gain * 100:.2f} pp)'
        ax.text(i, f1 * 100 + 0.8, label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "baseline_f1.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"F1 plot saved to {plot_path}")


def run_baseline_study():
    """执行基线模型对比实验"""
    logger.info("="*60)
    logger.info("基线模型对比实验")
    logger.info("="*60)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    # 遍历每个模型配置
    for model_key, model_config in BASELINE_CONFIGS.items():
        result = train_and_evaluate(model_key, model_config)
        all_results.append(result)

        # 绘制混淆矩阵
        plot_confusion_matrix(
            np.array(result["y_true"]),
            np.array(result["y_pred"]),
            model_key,
            model_config["name"]
        )

        # 保存单个结果
        result_path = OUTPUT_DIR / f"result_{model_key}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

    # 绘制汇总对比图
    plot_summary_comparison(all_results)

    # 生成汇总表格
    logger.info("\n" + "="*60)
    logger.info("汇总表格")
    logger.info("="*60)

    # 创建 DataFrame
    df = pd.DataFrame(all_results)

    # 打印表格
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')

    # 输出到终端
    print("\n" + "="*80)
    print("基线模型对比结果汇总")
    print("="*80)

    summary_df = df[["model_name", "accuracy", "precision", "recall", "f1_score"]]
    print(summary_df.to_string(index=False))

    # 保存为 CSV
    csv_path = OUTPUT_DIR / "baseline_study_results.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    logger.info(f"\nResults saved to {csv_path}")

    # 保存为 JSON
    json_path = OUTPUT_DIR / "baseline_study_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    logger.info("\n基线模型对比实验完成！")

    return all_results


if __name__ == "__main__":
    run_baseline_study()
