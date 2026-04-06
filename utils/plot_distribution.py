# plot_distribution.py
"""
数据分布可视化脚本
===================

用途：
    读取 7 个静态测站（P2~P8）的 GNSS 观测 CSV 数据，
    绘制 LOS 与 NLOS 信号在三个关键特征上的分布对比直方图，
    用于分析不同类别在特征空间中的区分度。

主要功能：
    1. 读取并合并多个测站的 CSV 数据
    2. 按测站位置（location 1~7）分组绘制
    3. 三个特征行：C/N0（载波噪声比）、Elevation（高度角）、Pseudorange_residual（伪距残差）
    4. 第四行：3D 特征空间可视化
    5. 每列叠加显示 NLOS（橙色）和 LOS（蓝色）的直方图分布

使用方式：
    # 直接运行
    python plot_distribution.py

    # 输出
    #   outputs/static_experiment/figures/distribution_LOS_NLOS.png

输入：
    - DATA_DIR 指向的目录下的 P2.csv ~ P8.csv 文件
    - 精确列名：GPS_Time(s), PRN, nSV, pseudorange, C/N0, Elevation, Azimuth,
      err_tropo, err_iono, sat_clock_error, Pseudorange_residual,
      Normalized_Pseudorange_residual, Pr_rate_consitency,
      Sat_pos_x, Sat_pos_y, Sat_pos_z, LOS/NLOS_label

输出：
    - 4 行 × 7 列的可视化矩阵
    - 保存为 outputs/static_experiment/figures/distribution_LOS_NLOS.png（300 DPI）
"""

import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ================== 1. 路径配置 ==================
# 以当前脚本所在目录为基准，避免硬编码绝对路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data for sharing_csv'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'static_experiment' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 精确列名（根据 CSV 文件表头）
CNO_COL   = 'C/N0'               # 载波噪声比
ELE_COL   = 'Elevation'           # 高度角
PRE_COL   = 'Pseudorange_residual'  # 伪距残差
LABEL_COL = 'LOS/NLOS_label'      # LOS/NLOS 标记
# =====================================================


# ================== 2. 加载预处理后的数据 ==================
# 从 CSV 加载并进行预处理过滤（与 static_preprocess.py 一致）
all_dfs = []
files = sorted(glob.glob(os.path.join(DATA_DIR, 'P[2-8].csv')))
for idx, f in enumerate(files, start=1):
    df_i = pd.read_csv(f, skipinitialspace=True)
    df_i['location'] = idx
    all_dfs.append(df_i)

df = pd.concat(all_dfs, ignore_index=True)

# 预处理过滤（与 static_preprocess.py 一致）
n_before = len(df)

# 1. 过滤 Pseudorange_residual 异常值
df = df[df[PRE_COL].abs() <= 100].copy()

# 2. 过滤 Pr_rate_consitency 异常值（9999 是无效值标记）
df = df[df['Pr_rate_consitency'] != 9999.0].copy()

n_after = len(df)
print(f'Data loaded: {len(df)} records (filtered {n_before - n_after} outliers)')

locations = sorted(df['location'].unique())
n_loc = len(locations)
print(f'Locations: {n_loc}')

# ================== 3. 绘制 4×7 分布图 ==================
fig = plt.figure(figsize=(3.5 * n_loc, 14))

# 创建子图网格：4行7列
# 第1-3行是直方图，第4行是3D图
gs = fig.add_gridspec(4, n_loc, height_ratios=[1, 1, 1, 1.8], hspace=0.4, wspace=0.3)


def plot_feature(row_idx, feature_col, feature_label, bins=40, xlim=None):
    for col_idx, loc in enumerate(locations):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        sub = df[df['location'] == loc]

        los  = sub[sub[LABEL_COL] ==  1][feature_col].dropna()
        nlos = sub[sub[LABEL_COL] == -1][feature_col].dropna()

        n_total = len(los) + len(nlos)
        pct_los = (len(los) / n_total * 100) if n_total > 0 else 0
        pct_nlos = (len(nlos) / n_total * 100) if n_total > 0 else 0

        # NLOS 放在下面（橙色），LOS 叠在上面（蓝色），图例显示百分比
        if len(nlos) > 0:
            ax.hist(nlos, bins=bins, color='orange', alpha=0.7,
                    label=f'NLOS({pct_nlos:.0f}%)')
        if len(los) > 0:
            ax.hist(los,  bins=bins, color='blue',   alpha=0.7,
                    label=f'LOS({pct_los:.0f}%)')

        if xlim is not None:
            ax.set_xlim(*xlim)

        if row_idx == 0:
            ax.set_title(f'Location {loc}', fontsize=11, fontweight='bold')

        if col_idx == 0:
            ax.set_ylabel('Count', fontsize=10)

        ax.set_xlabel(feature_label, fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # 每张图都显示 LOS/NLOS 百分比图例
        ax.legend(fontsize=8, loc='upper right')


# 第 1 行：C/N0 分布
plot_feature(
    row_idx=0,
    feature_col=CNO_COL,
    feature_label='C/N0 (dB)',
    bins=30,
    xlim=(0, 60)
)

# 第 2 行：Elevation 分布
plot_feature(
    row_idx=1,
    feature_col=ELE_COL,
    feature_label='Elevation (deg)',
    bins=50,
    xlim=(0, 100)
)

# 第 3 行：Pseudorange_residual 分布
plot_feature(
    row_idx=2,
    feature_col=PRE_COL,
    feature_label='Pseudorange Residual (m)',
    bins=100,
    xlim=(-100, 100)
)


# 第 4 行：3D 特征空间可视化（每个地点一个子图）
MAX_SAMPLES_PER_LOC = 1000  # 每个地点最多采样点数

for col_idx, loc in enumerate(locations):
    ax_3d = fig.add_subplot(gs[3, col_idx], projection='3d')
    
    # 获取该地点的数据
    loc_df = df[df['location'] == loc]
    
    # 采样
    if len(loc_df) > MAX_SAMPLES_PER_LOC:
        loc_df_3d = loc_df.sample(n=MAX_SAMPLES_PER_LOC, random_state=42)
    else:
        loc_df_3d = loc_df.copy()
    
    # 分离 LOS 和 NLOS
    los_3d = loc_df_3d[loc_df_3d[LABEL_COL] == 1]
    nlos_3d = loc_df_3d[loc_df_3d[LABEL_COL] == -1]
    
    n_total = len(los_3d) + len(nlos_3d)
    pct_los = (len(los_3d) / n_total * 100) if n_total > 0 else 0
    pct_nlos = (len(nlos_3d) / n_total * 100) if n_total > 0 else 0
    
    # 绘制 NLOS 点（红色）
    if len(nlos_3d) > 0:
        ax_3d.scatter(
            nlos_3d[ELE_COL],
            nlos_3d[CNO_COL],
            nlos_3d[PRE_COL],
            c='red',
            alpha=0.6,
            label=f'NLOS ({pct_nlos:.0f}%)',
            s=15,
            marker='o'
        )
    
    # 绘制 LOS 点（绿色）
    if len(los_3d) > 0:
        ax_3d.scatter(
            los_3d[ELE_COL],
            los_3d[CNO_COL],
            los_3d[PRE_COL],
            c='green',
            alpha=0.6,
            label=f'LOS ({pct_los:.0f}%)',
            s=15,
            marker='o'
        )
    
    # 设置坐标轴标签
    ax_3d.set_xlabel('Elevation (deg)', fontsize=9, labelpad=5)
    ax_3d.set_ylabel('C/N0 (dB)', fontsize=9, labelpad=5)
    ax_3d.set_zlabel('Residual (m)', fontsize=9, labelpad=5)
    
    # 设置标题
    ax_3d.set_title(f'Location {loc}', fontsize=10, fontweight='bold')
    
    # 设置图例
    ax_3d.legend(fontsize=7, loc='upper left')
    
    # 设置视角
    ax_3d.view_init(elev=25, azim=45)


plt.suptitle('Feature Distribution & 3D Feature Space by Location (LOS vs NLOS)',
             fontsize=14, fontweight='bold', y=0.98)

# 保存组合图片
output_path = OUTPUT_DIR / 'distribution_LOS_NLOS.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Combined image saved to: {output_path}')


# ================== 4. 保存每张子图 ==================
# 单独保存直方图（每个地点每个特征）
print('\nSaving individual subplots...')

# 重新绘制并保存每个直方图子图
for row_idx, (feature_col, feature_label, bins, xlim) in enumerate([
    (CNO_COL, 'C/N0 (dB)', 30, (0, 60)),
    (ELE_COL, 'Elevation (deg)', 50, (0, 100)),
    (PRE_COL, 'Pseudorange Residual (m)', 100, (-100, 100))
]):
    for col_idx, loc in enumerate(locations):
        fig_single, ax_single = plt.subplots(figsize=(5, 4))
        
        sub = df[df['location'] == loc]
        los = sub[sub[LABEL_COL] == 1][feature_col].dropna()
        nlos = sub[sub[LABEL_COL] == -1][feature_col].dropna()
        
        n_total = len(los) + len(nlos)
        pct_los = (len(los) / n_total * 100) if n_total > 0 else 0
        pct_nlos = (len(nlos) / n_total * 100) if n_total > 0 else 0
        
        if len(nlos) > 0:
            ax_single.hist(nlos, bins=bins, color='orange', alpha=0.7,
                           label=f'NLOS ({pct_nlos:.0f}%)')
        if len(los) > 0:
            ax_single.hist(los, bins=bins, color='blue', alpha=0.7,
                           label=f'LOS ({pct_los:.0f}%)')
        
        ax_single.set_xlim(*xlim)
        ax_single.set_title(f'Location {loc} - {feature_label}', fontsize=12, fontweight='bold')
        ax_single.set_xlabel(feature_label, fontsize=10)
        ax_single.set_ylabel('Count', fontsize=10)
        ax_single.legend(fontsize=9)
        ax_single.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 保存子图
        feature_name = feature_col.replace('/', '_').replace(' ', '_')
        sub_output_path = OUTPUT_DIR / f'hist_{feature_name}_location{loc}.png'
        plt.savefig(sub_output_path, dpi=150, bbox_inches='tight')
        plt.close(fig_single)


# 单独保存每个地点的 3D 图
for col_idx, loc in enumerate(locations):
    fig_3d_single = plt.figure(figsize=(8, 6))
    ax_3d_single = fig_3d_single.add_subplot(111, projection='3d')
    
    loc_df = df[df['location'] == loc]
    
    if len(loc_df) > MAX_SAMPLES_PER_LOC:
        loc_df_3d = loc_df.sample(n=MAX_SAMPLES_PER_LOC, random_state=42)
    else:
        loc_df_3d = loc_df.copy()
    
    los_3d = loc_df_3d[loc_df_3d[LABEL_COL] == 1]
    nlos_3d = loc_df_3d[loc_df_3d[LABEL_COL] == -1]
    
    n_total = len(los_3d) + len(nlos_3d)
    pct_los = (len(los_3d) / n_total * 100) if n_total > 0 else 0
    pct_nlos = (len(nlos_3d) / n_total * 100) if n_total > 0 else 0
    
    if len(nlos_3d) > 0:
        ax_3d_single.scatter(
            nlos_3d[ELE_COL], nlos_3d[CNO_COL], nlos_3d[PRE_COL],
            c='red', alpha=0.6, label=f'NLOS ({pct_nlos:.0f}%)', s=15
        )
    
    if len(los_3d) > 0:
        ax_3d_single.scatter(
            los_3d[ELE_COL], los_3d[CNO_COL], los_3d[PRE_COL],
            c='green', alpha=0.6, label=f'LOS ({pct_los:.0f}%)', s=15
        )
    
    ax_3d_single.set_xlabel('Elevation (deg)', fontsize=11)
    ax_3d_single.set_ylabel('C/N0 (dB)', fontsize=11)
    ax_3d_single.set_zlabel('Residual (m)', fontsize=11)
    ax_3d_single.set_title(f'Location {loc} - 3D Feature Space', fontsize=12, fontweight='bold')
    ax_3d_single.legend(fontsize=10)
    ax_3d_single.view_init(elev=25, azim=45)
    
    # 保存 3D 子图
    sub_output_path = OUTPUT_DIR / f'3d_feature_space_location{loc}.png'
    plt.savefig(sub_output_path, dpi=150, bbox_inches='tight')
    plt.close(fig_3d_single)

print('All individual subplots saved!')

plt.show()
