# plot_training_history.py
"""
根据保存的训练历史数据重新绘制训练曲线。

使用方式:
    python -m work.plot_training_history

模式自动从 modules/constants.py 的 DEFAULT_SPLIT_MODE 读取。
"""

import sys
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 添加 stca 目录到路径
STCA_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(STCA_DIR))

from utils.logger_config import setup_logger
from data_loading.constants import DEFAULT_SPLIT_MODE

logger = setup_logger(__name__)


def load_data(split_mode, data_dir="outputs/stca"):
    """Load training history data for the specified split mode."""
    data_dir = Path(data_dir)

    # 优先加载 JSON 格式（不带时间戳的最新文件）
    json_path = data_dir / f"training_history_data_{split_mode}.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded data from JSON: {json_path}")
        return data

    # 备选加载 pickle 格式（不带时间戳的最新文件）
    pkl_path = data_dir / f"training_history_data_{split_mode}.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            history = pickle.load(f)
        data = {
            "epochs": list(range(1, len(history['train_loss']) + 1)),
            "train_loss": [float(x) for x in history['train_loss']],
            "val_loss": [float(x) for x in history['val_loss']],
            "train_acc": [float(x) for x in history['train_acc']],
            "val_acc": [float(x) for x in history['val_acc']],
            "train_f1": [float(x) for x in history['train_f1']],
            "val_f1": [float(x) for x in history['val_f1']],
            "split_mode": split_mode,
        }
        logger.info(f"Loaded data from pickle: {pkl_path}")
        return data

    # 兼容旧格式：查找带时间戳的文件
    import glob
    json_pattern = str(data_dir / f"training_history_data_{split_mode}_*.json")
    json_files = glob.glob(json_pattern)
    if json_files:
        # 选择最新的文件（按修改时间排序）
        latest_json = max(json_files, key=lambda p: Path(p).stat().st_mtime)
        with open(latest_json, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded data from JSON (legacy format): {latest_json}")
        return data

    raise FileNotFoundError(
        f"No data file found for split_mode '{split_mode}'.\n"
        f"Expected: {json_path} or {pkl_path}\n"
        f"Please run training first to generate the data file."
    )


def plot_from_data(plot_data, output_dir):
    """Generate plots from loaded data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_mode = plot_data.get("split_mode", "unknown")
    epochs = plot_data["epochs"]

    # 设置样式
    plt.style.use('seaborn-v0_8-whitegrid')

    # 文件名（不带时间戳）
    loss_filename = f"training_loss_{split_mode}.png"
    f1_filename = f"training_f1_{split_mode}.png"
    acc_filename = f"training_accuracy_{split_mode}.png"
    history_filename = f"training_history_{split_mode}.png"

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, plot_data['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(epochs, plot_data['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / loss_filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training loss plot saved to {output_dir / loss_filename}")

    # Plot F1 Score
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, plot_data['train_f1'], label='Training F1', linewidth=2)
    plt.plot(epochs, plot_data['val_f1'], label='Validation F1', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score (%)', fontsize=12)
    plt.title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f1_filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training F1 plot saved to {output_dir / f1_filename}")

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, plot_data['train_acc'], label='Training Accuracy', linewidth=2)
    plt.plot(epochs, plot_data['val_acc'], label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / acc_filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training accuracy plot saved to {output_dir / acc_filename}")

    # Combined plot - 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss subplot
    ax1 = axes[0]
    ax1.plot(epochs, plot_data['train_loss'], label='Training Loss', linewidth=2)
    ax1.plot(epochs, plot_data['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # F1 subplot
    ax2 = axes[1]
    ax2.plot(epochs, plot_data['train_f1'], label='Training F1', linewidth=2)
    ax2.plot(epochs, plot_data['val_f1'], label='Validation F1', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('F1 Score (%)', fontsize=11)
    ax2.set_title('F1 Score', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Accuracy subplot
    ax3 = axes[2]
    ax3.plot(epochs, plot_data['train_acc'], label='Training Accuracy', linewidth=2)
    ax3.plot(epochs, plot_data['val_acc'], label='Validation Accuracy', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Accuracy', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Training History', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / history_filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Combined training history plot saved to {output_dir / history_filename}")


def main():
    # 从 constants 读取默认模式
    split_mode = DEFAULT_SPLIT_MODE
    logger.info(f"Using split_mode from constants: {split_mode}")

    # 加载数据
    plot_data = load_data(split_mode)

    # 绘图（输出到 repaints 目录）
    output_dir = Path("outputs/stca/repaints")
    plot_from_data(plot_data, output_dir)

    logger.info("Replotting complete!")


if __name__ == "__main__":
    main()
