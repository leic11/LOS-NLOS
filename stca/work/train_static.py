# train_static.py
"""
静态数据集训练与评估脚本 - PyTorch Version
==================

用途：
    整合数据加载、模型构建、训练执行、测试评估、结果保存的端到端入口，
    适用于静态点 GNSS NLOS 信号的二分类任务。

主要功能：
    1. 配置管理：从 modules/constants.py 读取默认参数，支持命令行参数覆盖
    2. 随机种子固定：确保实验可复现
    3. 数据加载：优先加载已处理的 .npz 文件，否则触发预处理流程
    4. 模型构建：根据配置实例化 STCAModel，支持可选的时序编码器和交叉注意力模块
    5. 训练执行：调用 STCAModel.fit() 方法执行训练
    6. 训练结果可视化：绘制并保存训练损失曲线、准确率曲线、F1 分数曲线
    7. 测试集评估：准确率、精确率、召回率、F1、ROC AUC、PR AUC
    8. 评估可视化：混淆矩阵、ROC 曲线、PR 曲线

使用方式：
    修改 stca/data_loading/constants.py 中的 DEFAULT_SPLIT_MODE：
    - "indomain": 域内划分（按样本比例划分）
    - "outdomain": 域外划分（按地点划分）

    # 训练并测试模型
    python -m work.train_static

输入：
    - CSV 数据目录（默认 "data for sharing_csv"）
    - 已处理的 .npz 文件（默认 "static_processed_indomain.npz"）
    - JSON 配置文件（可选）

输出：
    - static_processed_indomain.npz：预处理后的数据集（如不存在）
    - outputs/stca/
        ├── final_model_{split_mode}.pth    # 完整模型权重
        ├── figures/
        │   ├── training_loss_{split_mode}.png
        │   ├── training_accuracy_{split_mode}.png
        │   ├── training_f1_{split_mode}.png
        │   ├── confusion_matrix_{split_mode}_{timestamp}.png
        │   ├── roc_curve_{split_mode}_{timestamp}.png
        │   └── pr_curve_{split_mode}_{timestamp}.png
        └── metrics.json             # 训练和测试指标
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径，以便导入 utils
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 添加 modules 目录到路径，以便导入 trainer、stca_model 等
MODULES_DIR = Path(__file__).parent / "modules"
sys.path.insert(0, str(MODULES_DIR))

# 添加 stca 目录到路径，以便导入 data_loading
STCA_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(STCA_DIR))

from utils.logger_config import setup_logger
from utils.seed_utils import set_seed
from modules.stca_model import STCAModel
from data_loading.main import StaticPreprocessor
import matplotlib.pyplot as plt
from pathlib import Path
import json
import matplotlib
import numpy as np
import torch
matplotlib.use('Agg')
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve, average_precision_score
)
import seaborn as sns

# 使用统一日志配置
logger = setup_logger(__name__)


def load_data(config):
    """Load or preprocess data for STCA model.

    STCA 模型始终使用双输入：
    - X_train_spatial: 空间输入 (samples, max_satellites, features)，用于 AAM 模块
    - X_train_temporal: 时间序列输入 (samples, window_size, features)，用于 LSTM 模块

    Returns:
        (X_train_spatial, X_train_temporal), (X_test_spatial, X_test_temporal), y_train, y_test
    """
    # 使用 stca 目录保存 .npz 文件
    script_dir = Path(__file__).parent.parent

    window_size = config.get("window_size", 10)
    split_mode = config.get("split_mode", "indomain")

    logger.info(f"Loading data with split_mode: {split_mode}")

    # Determine which npz file to use based on split_mode
    if split_mode == "outdomain":
        npz_filename = config.get(
            "processed_npz_outdomain", "static_processed_outdomain.npz")
    else:
        npz_filename = config.get(
            "processed_npz_indomain", "static_processed_indomain.npz")

    npz_path = script_dir / npz_filename

    if npz_path.exists():
        logger.info(f"Loading preprocessed data from {npz_path}")
        data = StaticPreprocessor.load_processed(str(npz_path))

        # 检查是否有 STCA 格式数据 (X_train_spatial + X_train_temporal)
        if data.get("X_train_spatial") is not None:
            # STCA 格式：空间输入 + 时间序列输入
            logger.info("Loading data with STCA format (spatial + temporal)")
            X_train_temporal = data["X_train_temporal"]
            X_test_temporal = data["X_test_temporal"]
            X_train_spatial = data["X_train_spatial"]
            X_test_spatial = data["X_test_spatial"]

            logger.info(
                f"  Spatial: Train {X_train_spatial.shape}, Test {X_test_spatial.shape}")
            logger.info(
                f"  Temporal: Train {X_train_temporal.shape}, Test {X_test_temporal.shape}")

            # 返回 (空间输入，时间输入), (测试集), y_train, y_test
            return (
                (X_train_spatial, X_train_temporal),
                (X_test_spatial, X_test_temporal),
                data["y_train"], data["y_test"]
            )
    else:
        logger.info(f"Processed data not found. Running preprocessing...")
        # 数据目录在 stca 的父目录 (DevLab) 下
        data_dir = Path(__file__).parent.parent.parent / config["data_dir"]
        preprocessor = StaticPreprocessor(
            data_dir=str(data_dir),
            test_size=config["test_size"],
            random_seed=config["random_seed"],
        )

        # 生成包含空间特征的数据
        logger.info("Generating data with spatial features for STCA model...")
        data = preprocessor.process_stca(
            window_size=window_size,
            split_mode=split_mode,
        )

        # 保存 npz，便于后续 eval 直接加载
        preprocessor.save_processed(data, str(npz_path))
        logger.info(f"Preprocessed data saved to {npz_path}")

        return (
            (data["X_train_spatial"], data["X_train_temporal"]),
            (data["X_test_spatial"], data["X_test_temporal"]),
            data["y_train"], data["y_test"]
        )



def plot_training_history(history, output_dir, split_mode="indomain"):
    """Plot and save training history curves."""
    import pickle
    import json

    figures_dir = Path(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 保存绘图数据（JSON 格式）
    plot_data = {
        "epochs": list(range(1, len(history['train_loss']) + 1)),
        "train_loss": [float(x) for x in history['train_loss']],
        "train_acc": [float(x) for x in history['train_acc']],
        "train_f1": [float(x) for x in history['train_f1']],
        "split_mode": split_mode,
    }

    # 保存为 JSON 格式（人类可读）- 覆盖旧文件
    plot_data_path = figures_dir / f"training_history_data_{split_mode}.json"
    with open(plot_data_path, 'w') as f:
        json.dump(plot_data, f, indent=4)
    logger.info(f"Training history data saved to {plot_data_path}")

    # 同时保存为 pkl 格式（保留完整精度）- 覆盖旧文件
    pkl_path = figures_dir / f"training_history_data_{split_mode}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(history, f)
    logger.info(f"Training history pickle saved to {pkl_path}")

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    epochs = plot_data["epochs"]

    # Filename without timestamp (overwrite on each run)
    loss_filename = f"training_loss_{split_mode}.png"
    f1_filename = f"training_f1_{split_mode}.png"
    acc_filename = f"training_accuracy_{split_mode}.png"
    history_filename = f"training_history_{split_mode}.png"

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_dir / loss_filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training loss plot saved to {figures_dir / loss_filename}")

    # Plot F1 Score
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_f1'], label='Training F1', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score (%)', fontsize=12)
    plt.title('Training F1 Score',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_dir / f1_filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training F1 plot saved to {figures_dir / f1_filename}")

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'],
             label='Training Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training Accuracy',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_dir / acc_filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(
        f"Training accuracy plot saved to {figures_dir / acc_filename}")


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

    # Create dataset
    class TensorDataset(torch.utils.data.Dataset):
        def __init__(self, x_2d_np, x_3d_np, y_np):
            self.x_2d = x_2d_np
            self.x_3d = x_3d_np
            self.y = y_np

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            x_2d_item = torch.from_numpy(self.x_2d[idx:idx+1].copy())
            x_3d_item = torch.from_numpy(self.x_3d[idx:idx+1].copy())
            y_item = torch.from_numpy(self.y[idx:idx+1].copy())
            return x_2d_item.squeeze(0), x_3d_item.squeeze(0), y_item.squeeze(0)

    dataset = TensorDataset(X_test_2d_np, X_test_3d_np, y_test_np)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 转换为训练脚本的格式
    test_loader = []
    for spatial_batch, temporal_batch, targets in loader:
        test_loader.append(((spatial_batch, temporal_batch), targets))

    return test_loader


def evaluate_model(model, test_loader, device, output_dir, split_mode="indomain"):
    """Evaluate model on test set and save metrics/plots."""
    logger.info("Evaluating model on test set...")

    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch_data in test_loader:
            if isinstance(batch_data[0], tuple):
                (data_2d, data_3d), targets = batch_data
            else:
                data_2d = batch_data[0]
                data_3d = batch_data[0]
                targets = batch_data[1]

            data_2d = data_2d.to(device)
            data_3d = data_3d.to(device)
            targets = targets.to(device)

            outputs = model(x_spatial=data_2d, x_temporal=data_3d)
            probs = outputs.squeeze(-1)
            preds = (probs >= 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

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
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

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

    logger.info("\n" + "="*50)
    logger.info("Test Set Evaluation Results")
    logger.info("="*50)
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"ROC AUC:   {roc_auc:.4f}")
    logger.info(f"PR AUC:    {pr_auc:.4f}")
    logger.info(f"\nClassification Report:\n{report}")

    # Save metrics to JSON (update existing metrics.json)
    metrics_path = Path(output_dir) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)
        all_metrics["test_metrics"] = metrics
    else:
        all_metrics = {"test_metrics": metrics}

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Test metrics saved to {metrics_path}")

    # Generate figures
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot 1: Confusion Matrix
    plt.figure(figsize=(8, 6))
    annot_labels = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_labels[i, j] = f'{cm[i, j]:d}\n{cm_percent[i, j]:.1f}%'

    sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap='Blues',
                cbar_kws={'format': '%.0f%%'},
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
    plt.legend(loc="lower right")
    plt.title('ROC Curve')
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

    logger.info("="*50)
    return metrics


from modules.constants import (
    SPATIAL_EMBED_DIM, SPATIAL_NUM_HEADS, SPATIAL_NUM_LAYERS, SPATIAL_D_FF, SPATIAL_DROPOUT,
    TEMPORAL_EMBED_DIM, TEMPORAL_NUM_LAYERS, TEMPORAL_DROPOUT, TEMPORAL_BIDIRECTIONAL,
    CROSS_ATTN_EMBED_DIM, CROSS_ATTN_NUM_HEADS, CROSS_ATTN_DROPOUT,
    CLASSIFIER_HIDDEN_DIMS, CLASSIFIER_DROPOUT,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_SEED,
)

# 数据预处理参数从 data_loading.constants 导入
from data_loading.constants import (
    DEFAULT_WINDOW_SIZE as WINDOW_SIZE,
    DEFAULT_MAX_SATELLITES as MAX_SATELLITES,
    DEFAULT_SPLIT_MODE as SPLIT_MODE,  # 从 constants 读取划分模式
    INPUT_DIM,
    NUM_CLASSES,
)

def main(args):
    # 从 constants 模块加载默认参数
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
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED,
        "window_size": WINDOW_SIZE,
        "max_satellites": MAX_SATELLITES,
        "input_dim": INPUT_DIM,
        "num_classes": NUM_CLASSES,
        "split_mode": SPLIT_MODE,  # 使用 constants 中配置的模式
        "data_dir": "data for sharing_csv",  # 原始 CSV 数据目录
        "test_size": 0.2,  # 测试集比例
        "output_dir": str(ROOT_DIR / "outputs" / "stca"),  # 统一到 DevLab/outputs/stca/
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Set seed
    set_seed(config["random_seed"])

    # Load data (always returns dual input format for STCA)
    data_result = load_data(config)

    # STCA 模型始终使用双输入 (空间 + 时间)
    # data_result is ((X_train_spatial, X_train_temporal), (X_test_spatial, X_test_temporal), y_train, y_test)
    (X_train_spatial, X_train_temporal), (X_test_spatial, X_test_temporal), y_train, y_test = data_result
    logger.info(f"Data loaded (STCA mode - spatial + temporal):")
    logger.info(
        f"  Spatial: Train {X_train_spatial.shape}, Test {X_test_spatial.shape}")
    logger.info(
        f"  Temporal: Train {X_train_temporal.shape}, Test {X_test_temporal.shape}")

    # Ensure input_dim matches data (spatial input is 2D: N, max_satellites, features)
    if config["input_dim"] != X_train_spatial.shape[-1]:
        logger.warning(
            f"Config input_dim ({config['input_dim']}) != Data feature dim ({X_train_spatial.shape[-1]}). Updating.")
        config["input_dim"] = X_train_spatial.shape[-1]

    # 标签分布检查
    print("\n=== [DEBUG] 标签分布检查 ===")
    print(f"训练集标签分布：NLOS={np.sum(y_train == 0)}, LOS={np.sum(y_train == 1)}, 比例={np.sum(y_train == 1) / len(y_train):.2%}")
    print(f"测试集标签分布：NLOS={np.sum(y_test == 0)}, LOS={np.sum(y_test == 1)}, 比例={np.sum(y_test == 1) / len(y_test):.2%}")

    # Create model
    logger.info("Building STCA model...")
    model = STCAModel(
        input_dim=config["input_dim"],
        num_classes=config.get("num_classes", 2),
    )

    # Print model summary
    logger.info("Model Architecture:")
    logger.info(str(model))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Train using model.fit() - 无验证集模式
    logger.info(f"使用学习率：{config['learning_rate']}")
    history = model.fit(
        X_train_spatial, y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        device=config["device"],
        verbose=True,
        X_train_temporal=X_train_temporal,
    )

    # Save final model (with split_mode in filename to avoid overwrite)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_name = f"final_model_{config['split_mode']}.pth"
    final_model_path = output_dir / final_model_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
    }, final_model_path)
    logger.info(f"Model saved to {final_model_path}")

    # Plot training history
    plot_training_history(history, config["output_dir"], config["split_mode"])

    # Evaluate on test set
    test_loader = prepare_test_loader(
        {"X_test_spatial": X_test_spatial, "X_test_temporal": X_test_temporal, "y_test": y_test},
        batch_size=64,
        device=config["device"]
    )
    test_metrics = evaluate_model(
        model, test_loader, config["device"],
        config["output_dir"], config["split_mode"]
    )

    logger.info("Training and evaluation complete!")


if __name__ == "__main__":
    logger.info(f"当前训练模式：{SPLIT_MODE}")
    main(None)
