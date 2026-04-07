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
├── requirements.txt               # Python 依赖
│
├── data for sharing_csv/          # 原始 GNSS 静态数据
│   └── P2.csv ~ P8.csv            # 各测站原始观测数据
│
├── stca/                          # 核心代码（扁平化结构）
│   ├── default_config.json        # 默认配置文件
│   ├── static_preprocess.py       # 数据预处理 (CSV → 特征工程 → npz)
│   ├── plot_distribution.py       # 数据分布可视化
│   ├── stca_model.py              # 完整 STCA 模型
│   ├── spatial_encoder.py         # 空间编码器 (AAM)
│   ├── temporal_encoder.py        # 时序编码器 (LSTM)
│   ├── cross_attention.py         # 交叉注意力模块
│   ├── sparse_representation.py   # 稀疏表示模块 (L1 正则化)
│   ├── dl_baselines.py            # 深度学习基线模型
│   ├── ablation_models.py         # 消融实验模型
│   ├── experiment_tracker.py      # 实验结果记录器
│   ├── trainer.py                 # 训练器类
│   ├── train_static.py            # 训练脚本
│   ├── eval_static.py             # 评估脚本
│   ├── ablation_window_size.py    # 窗口大小消融实验
│   └── seed_utils.py              # 随机种子工具
│
├── outputs/                       # 实验输出目录（保留在根目录）
│   └── stca/                      # STCA 实验结果
│       ├── ws6/, ws8/, ws10/...   # 不同窗口大小实验
│       └── ablation_window_size/  # 窗口消融结果
│
└── utils/                         # 工具模块（保留在根目录）
    └── seed_utils.py              # 随机种子设置
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 进入 stca 目录
cd stca

# 使用默认配置训练
python -m train_static

# 指定配置文件
python -m train_static --config default_config.json
```

### 3. 评估模型

```bash
# 评估模型
python -m eval_static
```

### 4. 窗口大小消融实验

```bash
python ablation_window_size.py
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
| err_tropo | 对流层延迟误差 | 米 |
| err_iono | 电离层延迟误差 | 米 |
| sat_clock_error | 卫星钟差 | 秒 |
| Pseudorange_residual | 伪距残差 | 米 |
| Normalized_Pseudorange_residual | 归一化伪距残差 | - |
| Pr_rate_consitency | 伪距变化率一致性 | - |
| Sat_pos_x/y/z | 卫星位置坐标 (ECEF) | 米 |
| LOS/NLOS_label | 标签：-1=NLOS, 1=LOS | - |

### 模型输入特征（4 个）

STCA 模型使用论文推荐的 4 个核心特征：

| 特征 | 说明 |
|------|------|
| C/N0 | 载波噪声比 (dB-Hz) |
| Pseudorange_residual | 伪距残差 (m) |
| Elevation | 高度角 (°) |
| Azimuth | 方位角 (°) |

### 标签

- `0` → NLOS (非视距信号)
- `1` → LOS (视距信号)

### 数据预处理流程

```
原始 CSV → 加载合并 → 过滤 (NaN/异常值) → 窗口化 → 数据集划分 → 归一化 → .npz
```

1. **加载合并**: 将所有测站 CSV 合并为一个 DataFrame，添加 `location` 列
2. **过滤**: 
   - 删除含 NaN 的行
   - 删除 `Pr_rate_consitency == 9999.0` 的无效数据
3. **窗口化**: 生成时空双通道输入（见下文）
4. **数据集划分**:
   - `indomain`: 内域划分 - 每个地点内独立随机划分训练/测试集，再从训练集划分验证集
   - `outdomain`: 跨域划分 - P2-P5 训练，P6 验证，P7-P8 测试（测试集地点在训练时未见）
5. **归一化**: 仅在训练集上拟合 scaler，然后应用到所有子集

---

### 数据集划分策略

STCA 支持两种数据划分模式，用于评估模型在不同场景下的泛化能力。

#### Indomain (内域) 划分

**目的**: 评估模型在**同一分布内**的泛化能力

**逻辑**:
```
1. 对每个地点独立进行随机划分
2. 每个地点内：70% 训练 / 30% 测试
3. 所有地点的训练集合并后，再划分 20% 作为验证集

例如 7 个地点 (P2-P8)，每个地点 1000 个样本：
  - P2: 700 训练 + 300 测试
  - P3: 700 训练 + 300 测试
  - ...
  - 总训练集：4900 → 划分出 20% 验证集 → 最终 Train: 3920, Val: 980, Test: 2100
```

**特点**:
- 训练集、验证集、测试集**都包含所有地点**的数据
- 模型在训练时"见过"所有地点
- 测试集样本来自训练时见过的地点（相同分布）

**适用场景**:
- 验证模型的基本学习能力
- 测试同一测站的新数据预测性能

---

#### Outdomain (跨域) 划分

**目的**: 评估模型对**未见地点**的域外泛化能力 (Out-of-Distribution Generalization)

**逻辑**:
```
1. 按地点完全分离划分
2. 固定划分规则:
   - P2, P3, P4, P8 → 训练集 (4 个地点)
   - P7           → 验证集 (1 个地点)
   - P5, P6       → 测试集 (2 个地点)
```

**特点**:
- 训练集、验证集、测试集的地点**完全不重叠**
- 模型在训练时**从未见过** P7、P8 的数据
- 测试集样本来自全新的地理位置（不同分布）

**适用场景**:
- 验证模型的跨地点泛化能力
- 测试模型能否推广到未部署过的测站
- 更符合实际应用需求（新地点无标注数据）

---

#### 两种划分对比

| 划分方式 | 训练地点 | 验证地点 | 测试地点 | 评估目标 |
|---------|---------|---------|---------|---------|
| Indomain | P2-P8 (部分样本) | P2-P8 (部分样本) | P2-P8 (部分样本) | 同分布内泛化 |
| Outdomain | P2, P3, P4, P8 | P7 | P5, P6 | 跨地点域外泛化 |

**为什么需要 Outdomain?**

GNSS NLOS 检测的实际挑战是：在一个地点训练的模型，能否直接应用到新的地理位置？
- 不同地点的卫星几何分布不同
- 不同地点的多径环境不同（高楼、山谷等）
- Outdomain 测试结果更能反映实际部署能力

---

### 时空双通道输入

STCA 模型采用双通道输入设计，同时捕获**时序特征**和**空间特征**。

#### 时间通道 (Temporal Input)

**目的**: 捕获单颗卫星的连续观测历史（时序特征）

**形状**: `(N, window_size, 4)` = `(N, 10, 4)`

**生成逻辑**:
```
1. 按 (location, PRN) 组合分组 - 不同地点的相同 PRN 视为独立卫星
2. 每个 (地点，PRN) 组合内的数据按 GPS_Time 排序
3. 滑动窗口切片，窗口大小=10
4. 每个窗口 = 该卫星连续 10 个历元的观测

例如卫星 PRN=2 在 P2 测站有 100 个历元：
  - 窗口 1: 历元 1-10   → 标签 = 历元 10 的 LOS/NLOS
  - 窗口 2: 历元 2-11   → 标签 = 历元 11 的 LOS/NLOS
  - ...
  - 窗口 91: 历元 91-100 → 标签 = 历元 100 的 LOS/NLOS

注意：PRN=2 在 P3 测站的观测会被独立处理，不会与 P2 的数据混合
```

**4 个特征**: `[C/N0, Elevation, Azimuth, Pseudorange_residual]`

---

#### 空间通道 (Spatial Input)

**目的**: 捕获同一时刻所有可见卫星的分布（空间特征）

**形状**: `(N, max_satellites, 4)` = `(N, 20, 4)`

**生成逻辑**:
```
1. 按 GPS_Time(s) 分组（同一时刻）
2. 每个时刻提取所有可见卫星的特征
3. 如果卫星数 < 20，用 0 填充到 20
4. 如果卫星数 > 20，只取前 20 颗（实际数据中单历元卫星数通常 ≤ 20）

例如 t=50418 时刻有 7 颗卫星：
  - sat_features.shape = (7, 4)
  - 填充后 = (20, 4)，后面 13 行全是 0
```

**关键点**: 空间通道的每个样本对应时间通道窗口的**最后一个历元**（窗口结束时刻）。

---

#### 双通道对齐

```
时间通道样本 i: [t-9, t-8, ..., t] 时刻的 PRN=X 观测  → shape (10, 4)
空间通道样本 i: t 时刻所有卫星的观测                  → shape (20, 4)
标签 i:        t 时刻 PRN=X 的 LOS/NLOS 标志
```

这样设计让模型能同时看到：
- **时间维度**: 这颗卫星过去 10 个历元信号质量如何变化
- **空间维度**: 同一时刻天空所有卫星的几何分布

## 🧠 模型架构

```
输入 → Spatial Encoder → Temporal Encoder → Cross-Attention → Sparse Rep → Classifier
       (AAM 模块)        (LSTM-TFE 模块)      (时空融合)        (L1 正则)    (MLP+Sigmoid)
```

### 模型配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| input_dim | 输入特征维度 | 4 |
| spatial_embed_dim | 空间嵌入维度 | 64 |
| temporal_embed_dim | 时间嵌入维度 | 64 |
| use_cross_attention | 是否使用交叉注意力 | true |
| sparse_weight | L1 正则化权重 | 1e-4 |
| batch_size | 批大小 | 16 |
| learning_rate | 学习率 | 1e-3 |

## 📚 依赖

- Python 3.10+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## 📖 引用

如果使用本代码，请引用原始论文：

```bibtex
@article{zeng2024stca,
  title={A Spatiotemporal Information-Driven Cross-Attention Model With Sparse Representation for GNSS NLOS Signal Detection},
  author={Zeng, et al.},
  year={2024}
}
```
