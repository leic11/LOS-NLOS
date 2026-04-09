# eval_static.py
"""
静态数据集模型评估脚本
======================

用途：
    对训练好的 PyTorch 模型在测试集上进行系统性评估，
    输出量化指标并生成可视化图表。

主要功能：
    1. 模型加载：使用 PyTorch 加载 .pth 权重文件
    2. 预测推理：在测试集上执行前向传播，获取预测概率
    3. 指标计算：准确率、精确率、召回率、F1 分数、ROC AUC
    4. 混淆矩阵：生成并保存热力图
    5. ROC 曲线：绘制并保存 ROC 曲线图
    6. PR 曲线：绘制并保存 Precision-Recall 曲线图

使用方式：
    修改 stca/data_loading/constants.py 中的 DEFAULT_SPLIT_MODE：
    - "indomain": 域内划分（按样本比例划分）
    - "outdomain": 域外划分（按地点划分）

    # 评估模型
    python -m work.eval_static
"""
import os
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve, average_precision_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root dir to path for imports (stca/)
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Add stca dir to path for data_loading imports
STCA_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(STCA_DIR))

# Add modules dir to path for stca_model imports
MODULES_DIR = Path(__file__).parent / "modules"
sys.path.insert(0, str(MODULES_DIR))

from utils.logger_config import setup_logger
from data_loading.main import StaticPreprocessor
from modules.stca_model import STCAModel
from data_loading.constants import DEFAULT_SPLIT_MODE as SPLIT_MODE  # 从 constants 读取划分模式


# 使用统一日志配置
logger = setup_logger(__name__)


def load_config_from_constants():
    """从 modules/constants.py 加载配置"""
    from modules.constants import (
        SPATIAL_EMBED_DIM, SPATIAL_NUM_HEADS, SPATIAL_NUM_LAYERS, SPATIAL_D_FF, SPATIAL_DROPOUT,
        TEMPORAL_EMBED_DIM, TEMPORAL_NUM_LAYERS, TEMPORAL_DROPOUT, TEMPORAL_BIDIRECTIONAL,
        CROSS_ATTN_EMBED_DIM, CROSS_ATTN_NUM_HEADS, CROSS_ATTN_DROPOUT,
        CLASSIFIER_HIDDEN_DIMS, CLASSIFIER_DROPOUT,
    )
    # 数据预处理参数从 data_loading.constants 导入
    from data_loading.constants import INPUT_DIM, NUM_CLASSES

    config = {
        "spatial_embed_dim": SPATIAL_EMBED_DIM,
        "spatial_num_heads": SPATIAL_NUM_HEADS,
        "spatial_num_layers": SPATIAL_NUM_LAYERS,
        "spatial_d_ff": SPATIAL_D_FF,
        "spatial_dropout": SPATIAL_DROPOUT,
        "temporal_embed_dim": TEMPORAL_EMBED_DIM,
        "temporal_num_layers": TEMPORAL_NUM_LAYERS,
        "temporal_dropout": TEMPORAL_DROPOUT,
        "temporal_bidirectional": TEMPORAL_BIDIRECTIONAL,
        "cross_attn_embed_dim": CROSS_ATTN_EMBED_DIM,
        "cross_attn_num_heads": CROSS_ATTN_NUM_HEADS,
        "cross_attn_dropout": CROSS_ATTN_DROPOUT,
        "classifier_hidden_dims": CLASSIFIER_HIDDEN_DIMS,
        "classifier_dropout": CLASSIFIER_DROPOUT,
        "num_classes": NUM_CLASSES,
        "input_dim": INPUT_DIM,
    }
    logger.info("Loaded config from modules/constants.py")
    return config


def load_model(model_path: str, input_dim: int, device: torch.device, config: dict = None):
    """Load PyTorch model with weights."""
    if config is None:
        config = load_config_from_constants()

    logger.info(f"Loading model from {model_path}")

    # Check if path exists
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract model state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Create model instance: config from constants.py
    model = STCAModel(
        input_dim=input_dim,
        num_classes=config.get("num_classes", 2),
        # Spatial encoder (AAM Module)
        spatial_embed_dim=config.get("spatial_embed_dim", 64),
        spatial_num_heads=config.get("spatial_num_heads", 1),
        spatial_num_layers=config.get("spatial_num_layers", 1),
        spatial_d_ff=config.get("spatial_d_ff", 128),
        spatial_dropout=config.get("spatial_dropout", 0.5),
        # Temporal encoder (LSTM-TFE Module)
        temporal_embed_dim=config.get("temporal_embed_dim", 64),
        temporal_num_layers=config.get("temporal_num_layers", 1),
        temporal_dropout=config.get("temporal_dropout", 0.5),
        temporal_bidirectional=config.get("temporal_bidirectional", False),
        # Cross attention
        cross_attn_embed_dim=config.get("cross_attn_embed_dim", 64),
        cross_attn_num_heads=config.get("cross_attn_num_heads", 1),
        cross_attn_dropout=config.get("cross_attn_dropout", 0.5),
        # Classifier
        classifier_hidden_dims=config.get("classifier_hidden_dims", [64, 32]),
        classifier_dropout=config.get("classifier_dropout", 0.3),
    )

    # Load weights (strict=True: 确保结构与 checkpoint 一致)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully.")
    return model


def load_data(npz_path: str):
    """Load preprocessed data."""
    logger.info(f"Loading data from {npz_path}")
    data = StaticPreprocessor.load_processed(npz_path)
    return data


def prepare_test_loader(data, batch_size: int = 64, device: torch.device = None):
    """Prepare test data loader for evaluation."""
    # Check if dual input (spatial + temporal) data is available
    if "X_test_spatial" in data and "X_test_temporal" in data:
        logger.info("Using dual input mode (spatial + temporal)")
        X_test_2d = data["X_test_spatial"]
        X_test_3d = data["X_test_temporal"]
    else:
        logger.info("Using single input mode (fallback)")
        X_test_2d = data["X_test"]
        X_test_3d = data["X_test_temporal"]

    y_test = data["y_test"]

    # 存储 numpy 数组，在 __getitem__ 中转为 tensor
    X_test_2d_np = np.array(X_test_2d, dtype=np.float32)
    X_test_3d_np = np.array(X_test_3d, dtype=np.float32)
    y_test_np = np.array(y_test, dtype=np.float32)

    # Create dataset - 返回 (x_2d, x_3d, y) 三元素元组，与训练一致
    class TensorDataset(torch.utils.data.Dataset):
        def __init__(self, x_2d_np, x_3d_np, y_np):
            self.x_2d = x_2d_np
            self.x_3d = x_3d_np
            self.y = y_np

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            # 用切片保持维度，避免索引返回标量
            x_2d_item = torch.from_numpy(self.x_2d[idx:idx+1].copy())
            x_3d_item = torch.from_numpy(self.x_3d[idx:idx+1].copy())
            y_item = torch.from_numpy(self.y[idx:idx+1].copy())
            # squeeze 去掉batch维度
            return x_2d_item.squeeze(0), x_3d_item.squeeze(0), y_item.squeeze(0)

    dataset = TensorDataset(X_test_2d_np, X_test_3d_np, y_test_np)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 转换为训练脚本的格式: [((spatial, temporal), labels), ...]
    test_loader = []
    for spatial_batch, temporal_batch, targets in loader:
        test_loader.append(((spatial_batch, temporal_batch), targets))

    return test_loader


def evaluate_model(model, test_loader, device, output_dir, split_mode="indomain"):
    """Evaluate model and save metrics/plots."""
    logger.info("Evaluating model...")

    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch_data in test_loader:
            # Handle both dual input and single input formats
            if isinstance(batch_data[0], tuple):
                (data_2d, data_3d), targets = batch_data
            else:
                data_2d = batch_data[0]
                data_3d = batch_data[0]  # Use same input for both
                targets = batch_data[1]

            data_2d = data_2d.to(device)
            data_3d = data_3d.to(device)
            targets = targets.to(device)

            # Forward pass（输出层已含 sigmoid，无需再套一层）
            outputs = model(x_spatial=data_2d, x_temporal=data_3d)
            probs = outputs.squeeze(-1)
            
            # Threshold at 0.5 for predictions
            preds = (probs >= 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy
    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)
    y_pred_prob = np.array(all_probs)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # PR Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = average_precision_score(y_true, y_pred_prob)

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=['NLOS', 'LOS'])

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }

    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Test Precision: {prec:.4f}")
    logger.info(f"Test Recall: {rec:.4f}")
    logger.info(f"Test F1: {f1:.4f}")
    logger.info(f"Test ROC AUC: {roc_auc:.4f}")
    logger.info(f"Test PR AUC: {pr_auc:.4f}")
    logger.info(f"\nClassification Report:\n{report}")

    # Save metrics
    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Create figures directory
    figures_dir = Path(output_dir) 
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot 1: Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NLOS', 'LOS'], yticklabels=['NLOS', 'LOS'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(figures_dir / f"confusion_matrix_{split_mode}_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {figures_dir / f'confusion_matrix_{split_mode}_{timestamp}.png'}")

    # Plot 2: ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.savefig(figures_dir / f"roc_curve_{split_mode}_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curve saved to {figures_dir / f'roc_curve_{split_mode}_{timestamp}.png'}")

    # Plot 3: PR Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(figures_dir / f"pr_curve_{split_mode}_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"PR curve saved to {figures_dir / f'pr_curve_{split_mode}_{timestamp}.png'}")

    return metrics


def main(args):
    # Load config from constants
    config = load_config_from_constants()

    # Defaults - 使用项目根目录作为基础路径
    root_dir = Path(__file__).parent.parent.parent
    split_mode = SPLIT_MODE  # 从 constants.py 读取
    npz_path = args.data or str(root_dir / "stca" / config.get(f"processed_npz_{split_mode}", f"static_processed_{split_mode}.npz"))
    # Default model path uses split_mode in filename
    default_model_name = f"final_model_{split_mode}.pth"
    model_path = args.model or str(root_dir / "outputs" / "stca" / default_model_name)
    output_dir = args.output or str(root_dir / "outputs" / "stca")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    data = load_data(npz_path)

    # Determine input dimension
    if "X_test_spatial" in data:
        input_dim = data["X_test_spatial"].shape[-1]
    else:
        input_dim = data["X_test"].shape[-1]
    logger.info(f"Input dimension: {input_dim}")

    # Load model (with config)
    model = load_model(model_path, input_dim, device, config)

    # Prepare test loader
    batch_size = args.batch_size or 64
    test_loader = prepare_test_loader(data, batch_size, device)

    # Evaluate
    metrics = evaluate_model(model, test_loader, device, output_dir, split_mode)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PyTorch STCA model on static data.")
    # config 参数已移除，使用 constants.py 中的常量
    parser.add_argument("--data", type=str, default=None,
                       help="Path to processed .npz file.")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to PyTorch model weights (.pth).")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for results.")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for evaluation.")
    args = parser.parse_args()

    logger.info(f"当前评估模式：{SPLIT_MODE}")
    main(args)
