# train_static.py
"""
静态数据集训练脚本 - PyTorch Version
==================

用途：
    整合数据加载、模型构建、训练执行、结果保存的端到端训练入口，
    适用于静态点 GNSS NLOS 信号的二分类任务。

主要功能：
    1. 配置管理：从 modules/constants.py 读取默认参数，支持命令行参数覆盖
    2. 随机种子固定：确保实验可复现
    3. 数据加载：优先加载已处理的 .npz 文件，否则触发预处理流程
    4. 模型构建：根据配置实例化 STCAModel，支持可选的时序编码器和交叉注意力模块
    5. 训练执行：调用 STCAModel.fit() 方法执行训练
    6. 结果可视化：绘制并保存训练损失曲线、准确率曲线、F1 分数曲线
    7. 快速评估：在测试集上输出损失、准确率和 F1 分数

使用方式：
    修改 stca/data_loading/constants.py 中的 DEFAULT_SPLIT_MODE：
    - "indomain": 域内划分（按样本比例划分）
    - "outdomain": 域外划分（按地点划分）

    # 训练模型
    python -m work.train_static

输入：
    - CSV 数据目录（默认 "data for sharing_csv"）
    - 已处理的 .npz 文件（默认 "static_processed_indomain.npz"）
    - JSON 配置文件（可选）

输出：
    - static_processed_indomain.npz：预处理后的数据集（如不存在）
    - outputs/static_experiment/
        ├── best_model.pth           # 最优模型权重
        ├── final_model.pth          # 完整模型
        ├── figures/
        │   ├── training_loss.png    # 损失曲线
        │   ├── training_accuracy.png # 准确率曲线
        │   └── training_history.png # 组合曲线
        └── metrics.json             # 评估指标
"""

import os
import sys
from pathlib import Path

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

# 使用统一日志配置
logger = setup_logger(__name__)


def load_data(config):
    """Load or preprocess data for STCA model.

    STCA 模型始终使用双输入：
    - X_train_spatial: 空间输入 (samples, max_satellites, features)，用于 AAM 模块
    - X_train_temporal: 时间序列输入 (samples, window_size, features)，用于 LSTM 模块

    Returns:
        (X_train_spatial, X_train_temporal), (X_val_spatial, X_val_temporal), (X_test_spatial, X_test_temporal),
        y_train, y_val, y_test
    """
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

        # 检查是否有新的 STCA 格式数据 (X_train_spatial + X_train_temporal)
        if data.get("X_train_spatial") is not None:
            # STCA 格式：空间输入 + 时间序列输入
            logger.info("Loading data with STCA format (spatial + temporal)")
            X_train_temporal = data["X_train_temporal"]
            X_val_temporal = data["X_val_temporal"]
            X_test_temporal = data["X_test_temporal"]
            X_train_spatial = data["X_train_spatial"]
            X_val_spatial = data["X_val_spatial"]
            X_test_spatial = data["X_test_spatial"]
            max_satellites = data.get(
                "max_satellites", X_train_spatial.shape[1])

            logger.info(
                f"  Spatial: Train {X_train_spatial.shape}, Val {X_val_spatial.shape}, Test {X_test_spatial.shape}")
            logger.info(
                f"  Temporal: Train {X_train_temporal.shape}, Val {X_val_temporal.shape}, Test {X_test_temporal.shape}")

            # 返回 (空间输入, 时间输入)
            return (
                (X_train_spatial, X_train_temporal),
                (X_val_spatial, X_val_temporal),
                (X_test_spatial, X_test_temporal),
                data["y_train"], data["y_val"], data["y_test"]
            )
    else:
        logger.info(f"Processed data not found. Running preprocessing...")
        data_dir = script_dir / config["data_dir"]
        preprocessor = StaticPreprocessor(
            data_dir=str(data_dir),
            test_size=config["test_size"],
            val_size=config["val_size"],
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
            (data["X_val_spatial"], data["X_val_temporal"]),
            (data["X_test_spatial"], data["X_test_temporal"]),
            data["y_train"], data["y_val"], data["y_test"]
        )



def plot_training_history(history, output_dir, split_mode="indomain"):
    """Plot and save training history curves with timestamp."""
    import pickle
    import json

    figures_dir = Path(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 保存绘图数据（JSON 格式）
    plot_data = {
        "epochs": list(range(1, len(history['train_loss']) + 1)),
        "train_loss": [float(x) for x in history['train_loss']],
        "val_loss": [float(x) for x in history['val_loss']],
        "train_acc": [float(x) for x in history['train_acc']],
        "val_acc": [float(x) for x in history['val_acc']],
        "train_f1": [float(x) for x in history['train_f1']],
        "val_f1": [float(x) for x in history['val_f1']],
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
    plt.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_dir / loss_filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training loss plot saved to {figures_dir / loss_filename}")

    # Plot F1 Score
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_f1'], label='Training F1', linewidth=2)
    plt.plot(epochs, history['val_f1'], label='Validation F1', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score (%)', fontsize=12)
    plt.title('Training and Validation F1 Score',
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
    plt.plot(epochs, history['val_acc'],
             label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Validation Accuracy',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_dir / acc_filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(
        f"Training accuracy plot saved to {figures_dir / acc_filename}")

    # Combined plot - 3 rows
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss subplot
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # F1 subplot
    ax2 = axes[1]
    ax2.plot(epochs, history['train_f1'], label='Training F1', linewidth=2)
    ax2.plot(epochs, history['val_f1'], label='Validation F1', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('F1 Score (%)', fontsize=11)
    ax2.set_title('F1 Score', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Accuracy subplot
    ax3 = axes[2]
    ax3.plot(epochs, history['train_acc'],
             label='Training Accuracy', linewidth=2)
    ax3.plot(epochs, history['val_acc'],
             label='Validation Accuracy', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Accuracy', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Training History', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / history_filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(
        f"Combined training history plot saved to {figures_dir / history_filename}")


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
        "output_dir": "outputs/stca",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Set seed
    set_seed(config["random_seed"])

    # Load data (always returns dual input format for STCA)
    data_result = load_data(config)

    # STCA 模型始终使用双输入 (空间 + 时间)
    # data_result is ((X_train_spatial, X_train_temporal), ...)
    (X_train_spatial, X_train_temporal), (X_val_spatial, X_val_temporal), (X_test_spatial,
                                                                           X_test_temporal), y_train, y_val, y_test = data_result
    logger.info(f"Data loaded (STCA mode - spatial + temporal):")
    logger.info(
        f"  Spatial: Train {X_train_spatial.shape}, Val {X_val_spatial.shape}, Test {X_test_spatial.shape}")
    logger.info(
        f"  Temporal: Train {X_train_temporal.shape}, Val {X_val_temporal.shape}, Test {X_test_temporal.shape}")

    # Ensure input_dim matches data (spatial input is 2D: N, max_satellites, features)
    if config["input_dim"] != X_train_spatial.shape[-1]:
        logger.warning(
            f"Config input_dim ({config['input_dim']}) != Data feature dim ({X_train_spatial.shape[-1]}). Updating.")
        config["input_dim"] = X_train_spatial.shape[-1]

    # 标签分布检查
    print("\n=== [DEBUG] 标签分布检查 ===")
    print(f"训练集标签分布：NLOS={np.sum(y_train == 0)}, LOS={np.sum(y_train == 1)}, 比例={np.sum(y_train == 1) / len(y_train):.2%}")
    print(f"验证集标签分布：NLOS={np.sum(y_val == 0)}, LOS={np.sum(y_val == 1)}, 比例={np.sum(y_val == 1) / len(y_val):.2%}")
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

    # Train using model.fit()
    logger.info(f"使用学习率：{config['learning_rate']}")
    history = model.fit(
        X_train_spatial, y_train,
        X_val_spatial=X_val_spatial, y_val=y_val,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        device=config["device"],
        verbose=True,
        X_train_temporal=X_train_temporal,
        X_val_temporal=X_val_temporal,
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

    # Evaluate using model.evaluate()
    metrics = model.evaluate(
        X_test_spatial, y_test,
        device=config["device"],
        X_test_3d=X_test_temporal,
    )

    # Save metrics - 转换 numpy 类型为 Python 原生类型
    def convert_to_python_type(obj):
        """递归转换 numpy 类型为 Python 原生类型"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_python_type(v) for v in obj]
        return obj

    metrics_serializable = {
        k: convert_to_python_type(v)
        for k, v in metrics.items()
        if k not in ['y_pred', 'y_prob', 'y_true']  # 排除大数组
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    logger.info(f"当前训练模式：{SPLIT_MODE}")
    main(None)
