# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GNSS NLOS (Non-Line-of-Sight) signal detection using a Spatiotemporal Cross-Attention (STCA) model. Based on Zeng et al. (2024) paper. Achieves ~99% accuracy on static dataset.

## Quick Commands

```bash
# All commands run from the stca directory
cd stca

# Data preprocessing
python -m data_loading.main --generate-stca
python -m data_loading.main --split-mode outdomain --generate-stca

# Training
python -m work.train_static
python -m work.train_static --split-mode outdomain

# Model evaluation
python -m work.eval_static

# Replot training history (from saved data)
python -m work.plot_training_history

# Ablation study
python ablation_window_size.py
```

## Architecture Overview

```
DevLab/
├── stca/                        # 核心代码目录
│   ├── modules/                   # 模型模块
│   │   ├── constants.py           # 默认参数配置
│   │   ├── stca_model.py          # STCA 完整模型
│   │   ├── spatial_encoder.py     # 空间编码器 (AAM)
│   │   ├── temporal_encoder.py    # 时序编码器 (LSTM)
│   │   ├── cross_attention.py     # 交叉注意力模块
│   │   ├── sparse_representation.py # 稀疏表示模块
│   │   └── trainer.py             # 训练器类（已弃用，使用 stca_model.py 的 fit()/evaluate()）
│   ├── data_loading/              # 数据预处理模块
│   │   ├── main.py                # 主预处理器 (StaticPreprocessor)
│   │   ├── constants.py           # 数据预处理参数
│   │   ├── loaders.py             # CSV 数据加载器
│   │   ├── filters.py             # 数据过滤
│   │   ├── splitters.py           # 数据集划分
│   │   ├── windowers.py           # 时间窗口生成
│   │   └── normalizers.py         # 特征标准化
│   ├── work/                      # 实验脚本
│   │   ├── train_static.py        # 训练脚本
│   │   ├── eval_static.py         # 评估脚本
│   │   └── plot_training_history.py  # 从保存数据重新绘图
│   └── __init__.py
├── outputs/                     # 实验输出
├── utils/                       # 工具模块
│   ├── seed_utils.py            # 随机种子设置
│   └── logger_config.py         # 日志配置
└── data for sharing_csv/        # 原始 CSV 数据
```

## Key Components

**STCA Model** (`stca/modules/stca_model.py`):

- Dual input: spatial `(B, max_sats, 4)` + temporal `(B, window, 4)`
- Modules: Spatial Encoder → Temporal Encoder → Cross-Attention → Sparse Rep → Classifier
- Output: probability in [0, 1] via sigmoid

**Preprocessor** (`stca/data_loading/main.py`):

- 4 features: C/N0, Elevation, Azimuth, Pseudorange_residual
- Split modes: `indomain` (per-location) or `outdomain` (cross-location)
- Saves to `.npz` with X_train, X_train_spatial, y_train, etc.

**Trainer** (`stca/modules/trainer.py`):

- BCELoss with sigmoid output
- Dual-input mode: `model(x_spatial=..., x_temporal=...)`
- Saves best model by validation accuracy

## Configuration

默认参数在 `stca/modules/constants.py` 中定义。关键参数：

- `SPATIAL_EMBED_DIM`: 64
- `TEMPORAL_EMBED_DIM`: 64
- `CROSS_ATTN_EMBED_DIM`: 64
- `SPARSE_EMBED_DIM`: 64
- `BATCH_SIZE`: 16
- `LEARNING_RATE`: 1e-3
- `SPATIAL_DROPOUT`: 0.5
- `TEMPORAL_DROPOUT`: 0.5

## Debugging

See `DEBUGGING_GUIDE.md` for detailed troubleshooting flowcharts covering:

- Data shape validation
- Module-by-module forward pass checks
- Gradient flow verification
- Loss convergence issues

## 注意

编辑文件时：强制指定每次只写入 100～200 行，然后使用 edits 自动接收模式完成编写。