# GNSS NLOS Detection using Spatiotemporal Cross-Attention (STCA)

本项目实现了一个基于 PyTorch 的时空交叉注意力（STCA）模型，用于 GNSS NLOS（非视距）信号检测。该模型基于论文：

> Zeng, et al. "A Spatiotemporal Information-Driven Cross-Attention Model With Sparse Representation for GNSS NLOS Signal Detection" (2024)

## ✨ 项目特点

- 🧠 **深度学习模型**：采用时空交叉注意力机制，结合稀疏表示进行 NLOS 检测
- 📊 **完整实验流程**：支持数据预处理、模型训练、评估和结果可视化
- 🎯 **高精度**：在静态数据集上达到 99%+ 准确率
- 🔄 **模型对比**：支持多种基线模型和消融实验
- 📜 **历史记录**：自动保存所有实验结果到 JSON 文件

## 📁 项目结构

```
DevLab/
├── README.md                      # 项目说明文档
├── CLAUDE.md                      # Claude Code 配置指南
├── requirements.txt               # Python 依赖
│
├── data for sharing_csv/          # 原始 GNSS 静态数据
│   └── P2.csv ~ P8.csv            # 7 个测站原始观测数据
│
├── stca/                          # 核心代码目录
│   ├── modules/                   # 模型模块
│   │   ├── constants.py           # 模型超参数配置
│   │   ├── stca_model.py          # STCA 完整模型（支持消融实验配置）
│   │   ├── spatial_encoder.py     # 空间编码器 (AAM)
│   │   ├── temporal_encoder.py    # 时序编码器 (LSTM-TFE)
│   │   ├── cross_attention.py     # 交叉注意力模块
│   │   └── sparse_representation.py # 稀疏表示模块
│   ├── data_loading/              # 数据预处理模块
│   │   ├── main.py                # 主预处理器 (StaticPreprocessor)
│   │   ├── constants.py           # 数据预处理参数
│   │   ├── loaders.py             # CSV 数据加载器
│   │   ├── filters.py             # 数据过滤
│   │   ├── splitters.py           # 数据集划分 (Indomain/Outdomain)
│   │   ├── windowers.py           # 时间窗口生成
│   │   └── normalizers.py         # 特征标准化
│   ├── work/                      # 实验脚本
│   │   ├── train_static.py        # 训练脚本
│   │   ├── eval_static.py         # 评估脚本
│   │   ├── ablation_window_size.py    # 窗口大小消融实验
│   │   ├── ablation_modules.py        # 模块消融实验 (Concat/CrossAttn/Both)
│   │   └── ablation_feature_comparison.py  # 特征对比实验 (4 特征 vs 9 特征)
│   └── utils/                     # 工具模块
│       ├── plot_training_history.py   # 重绘训练历史
│       └── replot_confusion_matrix.py # 重绘混淆矩阵
│
├── basemodel/                     # 基础模型框架（衍生特征工程）
│   ├── config.py                  # 配置管理
│   ├── data.py                    # 数据加载
│   ├── model.py                   # 模型定义
│   ├── train.py                   # 训练循环
│   └── plotting.py                # 可视化
│
├── utils/                         # 通用工具
│   ├── logger_config.py           # 日志配置
│   ├── seed_utils.py              # 随机种子设置
│   └── plot_distribution.py       # 数据分布可视化
│
└── outputs/                       # 实验输出
    └── stca/
        ├── ablation/
        │   ├── window_size/       # 窗口大小消融结果
        │   ├── modules/           # 模块消融结果
        │   └── feature/           # 特征对比结果
        └── models/                # 保存的模型权重
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
cd stca

# 生成 STCA 格式数据（默认 indomain 划分，窗口大小=8）
python -m data_loading.main --generate-stca

# 生成 Outdomain 划分数据
python -m data_loading.main --split-mode outdomain --generate-stca
```

### 3. 训练模型

```bash
# 使用 indomain 数据训练
python -m work.train_static

# 使用 outdomain 数据训练
python -m work.train_static --split-mode outdomain
```

### 4. 评估模型

```bash
# 评估 indomain 模型
python -m work.eval_static

# 评估 outdomain 模型
python -m work.eval_static --split-mode outdomain
```

### 5. 消融实验

```bash
# 窗口大小消融实验（6, 8, 10, ..., 32）
python -m work.ablation_window_size

# 模块消融实验（Baseline/+CrossAttn/+Both）
python -m work.ablation_modules

# 特征对比实验（4 特征 vs 9 特征）
python -m work.ablation_feature_comparison
```

## 📊 数据说明

### 原始数据文件

数据来源：`data for sharing_csv/` 目录，包含 7 个测站的静态 GNSS 观测数据：

| 文件 | 测站 | 记录数 |
|------|------|--------|
| P2.csv | P2 | ~6,500 |
| P3.csv | P3 | ~8,500 |
| P4.csv | P4 | ~12,500 |
| P5.csv | P5 | ~15,000 |
| P6.csv | P6 | ~10,000 |
| P7.csv | P7 | ~12,500 |
| P8.csv | P8 | ~17,000 |

### 原始数据字段（17 列）

| 字段 | 说明 | 单位 |
|------|------|------|
| GPS_Time | GPS 时间戳 | 秒 |
| PRN | 卫星伪随机噪声码编号 | - |
| nSV | 可见卫星数量 | - |
| pseudorange | 伪距观测值 | 米 |
| C/N0 | 载波噪声比 | dB-Hz |
| Elevation | 卫星高度角 | 度 (°) |
| Azimuth | 卫星方位角 | 度 (°) |
| Pseudorange_residual | 伪距残差 | 米 |
| LOS/NLOS_label | 标签：-1=NLOS, 1=LOS | - |

### 模型输入特征

**4 个原始特征**（论文推荐）：
| 特征 | 说明 |
|------|------|
| C/N0 | 载波噪声比 (dB-Hz) |
| Elevation | 高度角 (°) |
| Azimuth | 方位角 (°) |
| Pseudorange_residual | 伪距残差 (m) |

**5 个衍生特征**（扩展至 9 特征）：
| 特征 | 说明 |
|------|------|
| Delta_CNR | CNR 一阶差分 |
| CNR_std | CNR 滑动标准差 |
| PrRes_std | 伪距残差标准差 |
| Delta_Elevation | 高度角变化率 |
| Delta_Azimuth | 方位角变化率 |

### 标签

- `0` → NLOS (非视距信号)
- `1` → LOS (视距信号)

## 🧠 模型架构

```
输入 → Spatial Encoder → Temporal Encoder → Cross-Attention → Sparse Rep → Classifier
       (AAM 模块)        (LSTM-TFE 模块)      (时空融合)        (L1 正则)    (MLP+Sigmoid)
```

### 模型配置参数

所有默认参数在 `stca/modules/constants.py` 中定义：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| SPATIAL_EMBED_DIM | 空间嵌入维度 | 16 |
| SPATIAL_NUM_HEADS | 注意力头数 | 4 |
| SPATIAL_NUM_LAYERS | 编码器层数 | 2 |
| SPATIAL_DROPOUT | 空间编码器 Dropout | 0.5 |
| TEMPORAL_EMBED_DIM | 时间嵌入维度 | 16 |
| TEMPORAL_NUM_LAYERS | LSTM 层数 | 2 |
| TEMPORAL_DROPOUT | 时间编码器 Dropout | 0.5 |
| CROSS_ATTN_EMBED_DIM | 交叉注意力维度 | 16 |
| CROSS_ATTN_NUM_HEADS | 交叉注意力头数 | 1 |
| CLASSIFIER_HIDDEN_DIMS | 分类器隐藏层 | [16, 8] |
| BATCH_SIZE | 批大小 | 16 |
| EPOCHS | 训练轮数 | 50 |
| LEARNING_RATE | 学习率 | 1e-4 |

## 📈 消融实验配置

### 1. 窗口大小消融 (`ablation_window_size.py`)

测试不同时间窗口长度对模型性能的影响：
- 窗口范围：6, 8, 10, ..., 32
- 使用 9 特征输入
- 输出：性能折线图和 JSON 结果

### 2. 模块消融 (`ablation_modules.py`)

验证核心模块的贡献，支持三种配置：

| 配置 | use_cross_attention | use_sparse_representation | 说明 |
|------|---------------------|---------------------------|------|
| **Baseline (Concat)** | `False` | `False` | 空间 + 时间特征直接拼接 |
| **+CrossAttn** | `True` | `False` | 添加交叉注意力融合 |
| **+Both (Proposed)** | `True` | `True` | 完整模型（交叉注意力 + 稀疏表示） |

**Baseline (Concat) 实现细节**：
```python
# 空间特征聚合：对 num_sats 维度取平均池化
spatial_pooled = spatial_emb.mean(dim=1)  # (batch, 64)

# Concat 拼接：[hₜ, hₛ] -> (batch, 128)
fused_features = torch.cat([temporal_emb, spatial_pooled], dim=-1)

# 直接送分类器
output = classifier(fused_features)
```

### 3. 特征对比 (`ablation_feature_comparison.py`)

对比 4 特征（原始）vs 9 特征（扩展）的性能差异。

## 📁 数据预处理流程

```
原始 CSV → 加载合并 → 过滤 → 衍生特征 → 窗口化 → 数据集划分 → 归一化 → .npz
```

1. **加载合并**: 将所有测站 CSV 合并为一个 DataFrame，添加 `location` 列
2. **过滤**: 删除含 NaN 的行和无效数据
3. **衍生特征**: 添加 Delta_CNR、CNR_std 等 5 个衍生特征
4. **窗口化**: 生成时空双通道输入
5. **数据集划分**: Indomain 或 Outdomain
6. **归一化**: 仅在训练集上拟合 scaler

### 时空双通道输入

**时间通道** `(N, window_size, 9)`：
- 单颗卫星连续 window_size 个历元的观测历史

**空间通道** `(N, max_satellites, 9)`：
- 同一时刻所有可见卫星的分布

**对齐方式**：
```
时间通道样本 i: [t-9, t-8, ..., t] 时刻的 PRN=X 观测
空间通道样本 i: t 时刻所有卫星的观测
标签 i:        t 时刻 PRN=X 的 LOS/NLOS 标志
```

## 📚 依赖

- Python 3.10+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## 📖 引用

如果使用本代码，请引用原始论文：

```bibtex
@article{zeng2024stca,
  title={A Spatiotemporal Information-Driven Cross-Attention Model With Sparse Representation for GNSS NLOS Signal Detection},
  author={Zeng, et al.},
  year={2024}
}
```
