"""
时间窗口消融实验可视化脚本
========================

绘制不同时间窗口大小下的模型性能对比图。

输出：
  outputs/figures/ablation_window_size.png

使用方式：
    python utils/plot_window_ablation.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "outputs" / "stca" / "ablation" / "window_size"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 窗口大小配置
WINDOW_SIZES = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

# 颜色配置
ACCURACY_COLOR = "#1f77b4"  # 蓝色
F1_COLOR = "#ff7f0e"        # 橙色


def load_results() -> dict:
    """加载所有窗口大小的实验结果"""
    results = {}
    for w in WINDOW_SIZES:
        json_path = DATA_DIR / f"result_w{w}.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                results[w] = json.load(f)
        else:
            print(f"Warning: {json_path} not found")

    print(f"Loaded {len(results)} results")
    for w, data in sorted(results.items()):
        print(f"  w={w}: Acc={data['accuracy']:.4f}, F1={data['f1_score']:.4f}")

    return results


def plot_window_ablation(results: dict) -> None:
    """绘制窗口大小消融实验结果"""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 准备数据
    windows = sorted(results.keys())
    accuracies = [results[w]["accuracy"] for w in windows]
    f1_scores = [results[w]["f1_score"] for w in windows]

    # 找到最优值
    best_acc_idx = accuracies.index(max(accuracies))
    best_f1_idx = f1_scores.index(max(f1_scores))

    # 左轴：Accuracy 和 F1 分数（折线图 + 标记点）
    line1 = ax1.plot(windows, accuracies, color=ACCURACY_COLOR,
                     marker='o', linewidth=2, markersize=6, label='Accuracy')
    ax1.scatter([windows[best_acc_idx]], [accuracies[best_acc_idx]],
                color=ACCURACY_COLOR, s=150, marker='*', zorder=5,
                label=f'Best Accuracy: {max(accuracies):.4f} (w={windows[best_acc_idx]})')

    ax1.plot(windows, f1_scores, color=F1_COLOR,
             marker='s', linewidth=2, markersize=6, label='F1 Score')
    ax1.scatter([windows[best_f1_idx]], [f1_scores[best_f1_idx]],
                color=F1_COLOR, s=150, marker='*', zorder=5,
                label=f'Best F1: {max(f1_scores):.4f} (w={windows[best_f1_idx]})')

    # 设置标签和标题
    ax1.set_xlabel('Window Size (epochs)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance Metric', fontsize=12, fontweight='bold')
    ax1.set_title('Time Window Ablation Study', fontsize=14, fontweight='bold')

    # 设置坐标轴范围
    ax1.set_xlim(min(windows) - 1, max(windows) + 1)
    ax1.set_ylim(min(min(accuracies), min(f1_scores)) - 0.02,
                 max(max(accuracies), max(f1_scores)) + 0.02)

    # 网格
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 图例
    ax1.legend(loc='lower right', fontsize=10)

    # 添加最优值标注
    plt.annotate(f'w={windows[best_f1_idx]}\nF1={max(f1_scores):.4f}',
                 xy=(windows[best_f1_idx], max(f1_scores)),
                 xytext=(windows[best_f1_idx] + 2, max(f1_scores) - 0.03),
                 fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=F1_COLOR, lw=2))

    plt.tight_layout()

    # 保存图片
    output_path = OUTPUT_DIR / "ablation_window_size.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    print("Loading window ablation results...")
    results = load_results()

    print("\nGenerating ablation plot...")
    plot_window_ablation(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
