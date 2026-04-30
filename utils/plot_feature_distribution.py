"""
GNSS 特征分布可视化脚本
========================

绘制整个数据集的 4 个特征在 LOS/NLOS 下的统计分布对比图。

布局：
  - 图1-3（C/N0、高度角、伪距残差）：2行4列 = 8 张子图
    前 7 张：各地点单独数据（P2~P8）
    第 8 张：所有地点合并数据
  - 图4（3D 特征空间）：使用 P5 地点数据绘制三维散点图

过滤规则（与 data_loading/filters.py 一致）：
  - 丢弃含 NaN 的行
  - Pr_rate_consitency != 9999.0
  - |Pseudorange_residual| < 100m

输出：
  outputs/figures/distribution_CN0.png
  outputs/figures/distribution_elevation.png
  outputs/figures/distribution_pr.png
  outputs/figures/distribution_3d.png

使用方式：
    python utils/plot_feature_distribution.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data for sharing_csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 地点列表（源数据：P2~P8，显示为：P1~P7）
LOCATION_FILES = ["P2", "P3", "P4", "P5", "P6", "P7", "P8"]
LOCATION_LABELS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]

# 3D 图使用的地点（使用源数据名称）
LOCATION_3D = "P5"
LOCATION_3D_LABEL = "P4"  # P5 对应显示为 P4

# 特征配置
FEATURES = [
    {
        "col": "C/N0",
        "label": "C/N0 (dB-Hz)",
        "bins": 30,
        "xlim": (0, 60),
        "filename": "distribution_CN0.png",
    },
    {
        "col": "Elevation",
        "label": "Elevation Angle (deg)",
        "bins": 50,
        "xlim": (0, 90),
        "filename": "distribution_elevation.png",
    },
    {
        "col": "Pseudorange_residual",
        "label": "Pseudorange Residual (m)",
        "bins": 100,
        "xlim": (-100, 100),
        "filename": "distribution_pr.png",
    },
]

# 过滤阈值（与 constants.py 一致）
PR_RATE_INVALID = 9999.0
PR_THRESHOLD = 100.0
LABEL_COL = "LOS/NLOS_label"

# 颜色
LOS_COLOR = "#1f77b4"
NLOS_COLOR = "#ff7f0e"

# 每个子图最大采样数（避免直方图卡顿）
MAX_SAMPLES = 5000


def load_and_filter() -> pd.DataFrame:
    """加载并过滤数据（与 DataFilter 一致）"""
    dfs = []
    for file_loc, label_loc in zip(LOCATION_FILES, LOCATION_LABELS):
        csv_path = DATA_DIR / f"{file_loc}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, skipinitialspace=True)
            df["location"] = label_loc  # 存储为显示名称
            dfs.append(df)
        else:
            print(f"Warning: {csv_path} not found, skipped")
     

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    df = pd.concat(dfs, ignore_index=True)
    n_before = len(df)

    # 过滤 1: 丢弃含 NaN 的行
    df = df.dropna()
    # 过滤 2: Pr_rate_consitency != 9999
    df = df[df["Pr_rate_consitency"] != PR_RATE_INVALID].copy()
    # 过滤 3: |Pseudorange_residual| < 100
    df = df[abs(df["Pseudorange_residual"]) < PR_THRESHOLD].copy()

    n_los = (df[LABEL_COL] == 1).sum()
    n_nlos = (df[LABEL_COL] == -1).sum()
    print(f"Final: {len(df)} records (LOS={n_los}, NLOS={n_nlos})")

    return df


def plot_2d_distribution(df: pd.DataFrame, config: dict) -> None:
    """绘制 2行4列 = 8 子图的 2D 分布对比"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    col = config["col"]
    loc_data = [df[df["location"] == loc] for loc in LOCATION_LABELS]
    loc_data.append(df)  # 第 8 个子图：合并数据

    titles = LOCATION_LABELS + ["All Locations"]

    for idx, (sub_df, title) in enumerate(zip(loc_data, titles)):
        ax = axes[idx]

        los_data = sub_df[sub_df[LABEL_COL] == 1][col].dropna()
        nlos_data = sub_df[sub_df[LABEL_COL] == -1][col].dropna()

        # 采样控制：按原始比例采样，保持类别分布不变
        n_total_raw = len(los_data) + len(nlos_data)
        if n_total_raw > MAX_SAMPLES * 2:
            # 按原始比例计算采样后的数量
            los_ratio = len(los_data) / n_total_raw
            nlos_ratio = len(nlos_data) / n_total_raw
            los_sample_size = int(MAX_SAMPLES * 2 * los_ratio)
            nlos_sample_size = int(MAX_SAMPLES * 2 * nlos_ratio)

            if len(los_data) > los_sample_size:
                los_data = los_data.sample(los_sample_size, random_state=42)
            if len(nlos_data) > nlos_sample_size:
                nlos_data = nlos_data.sample(nlos_sample_size, random_state=42)

        n_total = len(los_data) + len(nlos_data)
        pct_los = len(los_data) / n_total * 100 if n_total > 0 else 0
        pct_nlos = len(nlos_data) / n_total * 100 if n_total > 0 else 0

        # 直方图叠加
        if len(nlos_data) > 0:
            ax.hist(nlos_data, bins=config["bins"], color=NLOS_COLOR,
                    alpha=0.7, label=f'NLOS ({pct_nlos:.1f}%)',
                    edgecolor='white', linewidth=0.5)
        if len(los_data) > 0:
            ax.hist(los_data, bins=config["bins"], color=LOS_COLOR,
                    alpha=0.7, label=f'LOS ({pct_los:.1f}%)',
                    edgecolor='white', linewidth=0.5)

        ax.set_xlim(config["xlim"])
        ax.set_xlabel(config["label"], fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.tick_params(axis='both', labelsize=9)

    # 隐藏第 9 个（不存在）的子图
    axes[7].set_visible(True)  # 第 8 个子图就是 merged，不用隐藏

    plt.suptitle(config["label"], fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = OUTPUT_DIR / config["filename"]
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_3d_distribution(df: pd.DataFrame) -> None:
    """绘制 P5 地点的 3D 特征空间散点图"""
    loc_df = df[df["location"] == LOCATION_3D_LABEL]

    # 采样控制
    MAX_3D_SAMPLES = 3000
    if len(loc_df) > MAX_3D_SAMPLES:
        loc_df = loc_df.sample(MAX_3D_SAMPLES, random_state=42)

    los_3d = loc_df[loc_df[LABEL_COL] == 1]
    nlos_3d = loc_df[loc_df[LABEL_COL] == -1]

    n_total = len(los_3d) + len(nlos_3d)
    pct_los = len(los_3d) / n_total * 100 if n_total > 0 else 0
    pct_nlos = len(nlos_3d) / n_total * 100 if n_total > 0 else 0

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if len(nlos_3d) > 0:
        ax.scatter(
            nlos_3d["Elevation"], nlos_3d["C/N0"], nlos_3d["Pseudorange_residual"],
            c=NLOS_COLOR, alpha=0.6, label=f'NLOS ({pct_nlos:.1f}%)', s=15, marker='o'
        )
    if len(los_3d) > 0:
        ax.scatter(
            los_3d["Elevation"], los_3d["C/N0"], los_3d["Pseudorange_residual"],
            c=LOS_COLOR, alpha=0.6, label=f'LOS ({pct_los:.1f}%)', s=15, marker='o'
        )

    ax.set_xlabel('Elevation (deg)', fontsize=11, labelpad=5)
    ax.set_ylabel('C/N0 (dB-Hz)', fontsize=11, labelpad=5)
    ax.set_zlabel('Pseudorange Residual (m)', fontsize=11, labelpad=5)
    ax.set_title(f'3D Feature Space - {LOCATION_3D_LABEL}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "distribution_3d.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main() -> None:
    print("Loading and filtering data...")
    df = load_and_filter()

    print("\nGenerating distribution plots...")
    for config in FEATURES:
        print(f"  Plotting {config['col']}...")
        plot_2d_distribution(df, config)

    print("  Plotting 3D distribution...")
    plot_3d_distribution(df)

    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
