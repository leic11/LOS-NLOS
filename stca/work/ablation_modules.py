# ablation_modules.py
"""
消融实验 2：核心模块验证实验
============================

用途：
    验证交叉注意力融合模块与稀疏正则化模块对 GNSS LOS/NLOS 分类
    模型性能的提升作用。

实验设置：
    在固定其余实验条件的前提下，依次引入两个核心模块：
    1. Baseline: 无交叉注意力、无稀疏正则化
    2. +CrossAttn: 有交叉注意力、无稀疏正则化
    3. +Sparse: 无交叉注意力、有稀疏正则化

    对内域 (indomain) 和跨域 (outdomain) 数据分别进行实验

输出：
    - 6 张混淆矩阵：
      - indomain_baseline.png
      - indomain_crossattn.png
      - indomain_sparse.png
      - outdomain_baseline.png
      - outdomain_crossattn.png
      - outdomain_sparse.png

使用方式：
    python -m work.ablation_modules
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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
# 实验配置
# ============================================================================

# 模型配置（固定超参数）
CONFIG = {
    "random_seed": 42,
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 0.001,
    "max_satellites": DEFAULT_MAX_SATELLITES,
    "window_size": DEFAULT_WINDOW_SIZE,
    # 模型参数
    "spatial_embed_dim": 64,
    "temporal_embed_dim": 64,
    "cross_attn_embed_dim": 64,
    "classifier_hidden_dims": [64, 32],
}

# 消融实验配置：3 种模块组合
# 验证交叉注意力和稀疏正则化的独立贡献及协同作用
MODULE_CONFIGS = {
    "baseline": {
        "name": "Baseline",
        "use_cross_attention": False,
        "use_sparse_representation": False,
    },
    "crossattn": {
        "name": "+CrossAttn",
        "use_cross_attention": True,
        "use_sparse_representation": False,
    },
    "both": {
        "name": "+Both",
        "use_cross_attention": True,
        "use_sparse_representation": True,
    },
}

# 数据划分模式
SPLIT_MODES = ["indomain", "outdomain"]

# 输出目录
OUTPUT_DIR = ROOT_DIR / "outputs" / "stca" / "ablation" / "modules"


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


def train_model(split_mode, module_key, module_config):
    """训练单个模型配置"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Split Mode: {split_mode} | Module: {module_config['name']}")
    logger.info(f"{'='*60}")

    # 设置随机种子
    set_seed(CONFIG["random_seed"])

    # 加载数据
    data = load_or_preprocess_data(split_mode, CONFIG["window_size"])

    X_train_spatial = data["X_train_spatial"]
    X_train_temporal = data["X_train_temporal"]
    y_train = data["y_train"]
    X_val_spatial = data["X_val_spatial"]
    X_val_temporal = data["X_val_temporal"]
    y_val = data["y_val"]
    X_test_spatial = data["X_test_spatial"]
    X_test_temporal = data["X_test_temporal"]
    y_test = data["y_test"]

    logger.info(f"Data: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

    # 构建模型（带模块控制参数）
    model = STCAModel(
        input_dim=4,
        num_classes=2,
        spatial_embed_dim=CONFIG["spatial_embed_dim"],
        temporal_embed_dim=CONFIG["temporal_embed_dim"],
        cross_attn_embed_dim=CONFIG["cross_attn_embed_dim"],
        classifier_hidden_dims=CONFIG["classifier_hidden_dims"],
        use_cross_attention=module_config["use_cross_attention"],
        use_sparse_representation=module_config["use_sparse_representation"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 训练
    logger.info(f"Training for {CONFIG['epochs']} epochs...")
    history = model.fit(
        X_train_spatial, y_train,
        X_val_spatial=X_val_spatial, y_val=y_val,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        lr=CONFIG["learning_rate"],
        device=device,
        verbose=True,
        X_train_temporal=X_train_temporal,
        X_val_temporal=X_val_temporal,
    )

    # 评估
    logger.info("Evaluating on test set...")
    metrics = model.evaluate(
        X_test_spatial, y_test,
        device=device,
        X_test_3d=X_test_temporal,
    )

    # 获取预测结果用于绘制混淆矩阵
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(0, len(X_test_spatial), 64):
            end_idx = min(i + 64, len(X_test_spatial))
            x_spatial_batch = torch.FloatTensor(X_test_spatial[i:end_idx]).to(device)
            x_temporal_batch = torch.FloatTensor(X_test_temporal[i:end_idx]).to(device)

            outputs = model(x_spatial=x_spatial_batch, x_temporal=x_temporal_batch)
            probs = outputs.squeeze(-1)
            preds = (probs >= 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_test[i:end_idx])

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)

    result = {
        "split_mode": split_mode,
        "module": module_key,
        "module_name": module_config["name"],
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "y_pred": y_pred.tolist(),
        "y_true": y_true.tolist(),
    }

    return result, history


def plot_confusion_matrix(y_true, y_pred, split_mode, module_key, module_name):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    # 计算百分比（按行归一化）
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(8, 6))

    # 创建自定义注释：数值 + 百分比
    annot_labels = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_labels[i, j] = f'{cm[i, j]:d}\n{cm_percent[i, j]:.2f}%'

    # 绘制热力图（使用归一化的混淆矩阵用于颜色条）
    sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap='Blues', cbar_kws={'format': '%.0f%%'},
                xticklabels=['NLOS', 'LOS'], yticklabels=['NLOS', 'LOS'])

    plt.title(f'Confusion Matrix - {split_mode.upper()} - {module_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    plot_path = OUTPUT_DIR / f"confusion_{split_mode}_{module_key}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix saved to {plot_path}")
    return cm


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("Ablation Study 2: Module Verification Experiment")
    logger.info("="*60)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for split_mode in SPLIT_MODES:
        for module_key, module_config in MODULE_CONFIGS.items():
            result, history = train_model(split_mode, module_key, module_config)
            all_results.append(result)

            # 绘制混淆矩阵
            plot_confusion_matrix(
                np.array(result["y_true"]),
                np.array(result["y_pred"]),
                split_mode,
                module_key,
                module_config["name"]
            )

            # 保存单个结果
            result_path = OUTPUT_DIR / f"result_{split_mode}_{module_key}.json"
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

            logger.info(f"[{split_mode}] {module_config['name']}: "
                       f"ACC={result['accuracy']:.4f}, PRE={result['precision']:.4f}, "
                       f"REC={result['recall']:.4f}, F1={result['f1_score']:.4f}")

    # 输出汇总表格
    logger.info("\n" + "="*60)
    logger.info("Summary Table")
    logger.info("="*60)
    logger.info(f"{'Split Mode':<15} {'Module':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    logger.info("-"*75)
    for r in all_results:
        logger.info(f"{r['split_mode']:<15} {r['module_name']:<15} {r['accuracy']*100:<12.2f} "
                   f"{r['precision']*100:<12.2f} {r['recall']*100:<12.2f} {r['f1_score']*100:<12.2f}")

    # 保存汇总数据
    summary = {
        "results": [
            {
                "split_mode": r["split_mode"],
                "module": r["module"],
                "module_name": r["module_name"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1_score": r["f1_score"],
            }
            for r in all_results
        ]
    }
    summary_path = OUTPUT_DIR / "ablation_modules_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    logger.info("\nAblation study 2 complete!")


if __name__ == "__main__":
    main()
