# train_static.py
"""
静态数据集训练脚本 - PyTorch Version
==================

用途：
    整合数据加载、模型构建、训练执行、结果保存的端到端训练入口，
    适用于静态点 GNSS NLOS 信号的二分类任务。

主要功能：
    1. 配置管理：从 default_config.json 读取默认参数，支持 JSON 配置文件覆盖
    2. 随机种子固定：确保实验可复现
    3. 数据加载：优先加载已处理的 .npz 文件，否则触发预处理流程
    4. 模型构建：根据配置实例化 STCAModel，支持可选的时序编码器和交叉注意力模块
    5. 数据集封装：转换为 torch.utils.data.DataLoader
    6. 训练执行：调用 Trainer 类执行训练，自动保存最优权重
    7. 结果可视化：绘制并保存训练损失曲线、准确率曲线
    8. 快速评估：在测试集上输出损失和准确率

使用方式：
    # 使用默认配置训练
    python -m train.train_static

    # 指定 JSON 配置文件
    python -m train.train_static --config my_config.json

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

from utils.seed_utils import set_seed
from trainer import Trainer
from stca_model import STCAModel
from static_preprocess import StaticPreprocessor
import matplotlib.pyplot as plt
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



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

        # 检查是否有新的 STCA 格式数据 (X_train + X_train_spatial)
        if "X_train_spatial" in data:
            # STCA 格式：空间输入 + 时间序列输入
            logger.info("Loading data with STCA format (spatial + temporal)")
            X_train_temporal = data["X_train"]
            X_val_temporal = data["X_val"]
            X_test_temporal = data["X_test"]
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


def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size, device, use_temporal=False):
    """Create PyTorch DataLoaders.

    Args:
        X_train, X_val, X_test: Feature arrays
            - If use_temporal=False: 2D arrays (samples, features)
            - If use_temporal=True: tuples of (X_2d, X_3d) for both spatial and temporal inputs
        y_train, y_val, y_test: Label arrays
        batch_size: Batch size
        device: Device to load tensors to
        use_temporal: Whether temporal model is used (determines data format)

    Returns:
        If use_temporal=False:
            train_loader, val_loader, test_loader (single input)
        If use_temporal=True:
            (train_loader_2d, train_loader_3d), (val_loader_2d, val_loader_3d), (test_loader_2d, test_loader_3d)
    """
    def create_tensors(X, y):
        """Convert numpy arrays to PyTorch tensors."""
        X_tensor = torch.FloatTensor(X)
        # BCELoss需要float32类型标签
        y_arr = np.array(y, dtype=np.float32)
        y_tensor = torch.FloatTensor(y_arr)
        return X_tensor, y_tensor

    if use_temporal:
        # Data is in tuple format: (X_2d, X_3d)
        X_train_2d, X_train_3d = X_train
        X_val_2d, X_val_3d = X_val
        X_test_2d, X_test_3d = X_test

        # 创建配对的数据集，转换为 Trainer 期望的格式: [((spatial, temporal), labels), ...]
        def create_dual_list(X_2d, X_3d, y, batch_size, shuffle):
            """创建 [(spatial_batch, temporal_batch), targets] 格式的列表"""
            X_2d_tensor = torch.FloatTensor(X_2d)
            X_3d_tensor = torch.FloatTensor(X_3d)
            y_arr = np.array(y, dtype=np.float32)
            y_tensor = torch.FloatTensor(y_arr)

            dataset = TensorDataset(X_2d_tensor, X_3d_tensor, y_tensor)
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle)

            # 转换为 Trainer 期望的格式: ((spatial, temporal), labels)
            result = []
            for batch in loader:
                spatial_batch, temporal_batch, targets = batch
                result.append(((spatial_batch, temporal_batch), targets))
            return result

        train_loader = create_dual_list(
            X_train_2d, X_train_3d, y_train, batch_size, True)
        val_loader = create_dual_list(
            X_val_2d, X_val_3d, y_val, batch_size, False)
        test_loader = create_dual_list(
            X_test_2d, X_test_3d, y_test, batch_size, False)

        return train_loader, val_loader, test_loader
    else:
        # Single 2D data
        train_dataset = TensorDataset(*create_tensors(X_train, y_train))
        val_dataset = TensorDataset(*create_tensors(X_val, y_val))
        test_dataset = TensorDataset(*create_tensors(X_test, y_test))

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


def plot_training_history(history, output_dir, split_mode="indomain"):
    """Plot and save training history curves with timestamp."""
    from datetime import datetime

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    figures_dir = Path(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    epochs = range(1, len(history['train_loss']) + 1)

    # Filename with timestamp and split_mode
    loss_filename = f"training_loss_{split_mode}_{timestamp}.png"
    f1_filename = f"training_f1_{split_mode}_{timestamp}.png"
    acc_filename = f"training_accuracy_{split_mode}_{timestamp}.png"
    history_filename = f"training_history_{split_mode}_{timestamp}.png"

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


def load_default_config():
    """从 default_config.json 加载默认配置"""
    import json
    config_path = Path(__file__).parent.parent / "default_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main(args):
    # Load config from default_config.json
    config = load_default_config()
    if args.config:
        import json
        with open(args.config, "r") as f:
            user_config = json.load(f)
        config.update(user_config)

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

    # Create dataloaders（单一数据集，包含配对的空间和时间数据）
    train_loader, val_loader, test_loader = create_dataloaders(
        (X_train_spatial, X_train_temporal), (X_val_spatial,
                                              X_val_temporal), (X_test_spatial, X_test_temporal),
        y_train, y_val, y_test,
        config["batch_size"], config["device"],
        use_temporal=True
    )

    # 在 train_static.py 中添加这个检查（在数据加载后）
    print("\n=== [DEBUG] 标签分布检查 ===")
    print(f"训练集标签分布: NLOS={np.sum(y_train == 0)}, LOS={np.sum(y_train == 1)}")
    print(f"验证集标签分布: NLOS={np.sum(y_val == 0)}, LOS={np.sum(y_val == 1)}")
    print(f"测试集标签分布: NLOS={np.sum(y_test == 0)}, LOS={np.sum(y_test == 1)}")

    # Create model
    logger.info("Building STCA model...")
    model = STCAModel(
        input_dim=config["input_dim"],
        num_classes=config.get("num_classes", 1),
        # Spatial encoder (AAM Module)
        spatial_embed_dim=config.get("spatial_embed_dim", 64),
        spatial_num_heads=config.get("spatial_num_heads", 1),
        spatial_num_layers=config.get("spatial_num_layers", 1),
        spatial_d_ff=config.get("spatial_d_ff", 128),
        spatial_dropout=config.get("spatial_dropout", 0.5),
        # Temporal encoder (LSTM-TFE Module) - always enabled
        temporal_embed_dim=config.get("temporal_embed_dim", 64),
        temporal_num_layers=config.get("temporal_num_layers", 1),
        temporal_dropout=config.get("temporal_dropout", 0.5),
        temporal_bidirectional=config.get("temporal_bidirectional", False),
        # Cross attention
        use_cross_attention=config.get("use_cross_attention", True),
        cross_attn_embed_dim=config.get("cross_attn_embed_dim", 64),
        cross_attn_num_heads=config.get("cross_attn_num_heads", 1),
        cross_attn_dropout=config.get("cross_attn_dropout", 0.5),
        # Classifier
        classifier_hidden_dims=config.get("classifier_hidden_dims", [64, 32]),
        classifier_dropout=config.get("classifier_dropout", 0.3),
    )

    # Print model summary
    logger.info("Model Architecture:")
    logger.info(str(model))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98))
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=config["device"],
        output_dir=config["output_dir"],
        use_sparse_loss=config.get("use_sparse_loss", True),  # 论文 3.3 节：L = L_BCE + λ·||h_fusion||₁
        sparse_weight=config.get("sparse_weight", 1e-4),
        use_dual_input=True,
        split_mode=config["split_mode"],
    )

    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=config["epochs"],
        early_stopping=False,
        verbose=True,
    )

    # Save final model (with split_mode in filename to avoid overwrite)
    final_model_name = f"final_model_{config['split_mode']}.pth"
    trainer.save_model(final_model_name)

    # Plot training history
    plot_training_history(history, config["output_dir"], config["split_mode"])

    # Evaluate
    metrics = trainer.evaluate(test_loader)
    trainer.save_metrics(metrics)

    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train STCA model on static GNSS NLOS data (PyTorch).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file.")
    args = parser.parse_args()
    main(args)
