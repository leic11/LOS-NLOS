# stca_model.py
"""
时空交叉注意力 (STCA) 模型（带稀疏表示）- PyTorch 版本
=============================================================

STCA模型是整个项目的核心模型，专门用于GNSS NLOS(非视距信号)检测任务。
它创新性地结合了空间特征提取、时间序列建模、交叉注意力机制和稀疏表示，
实现了对GNSS信号的多维度感知和准确分类。

模型设计理念:
    1. 空间优先: 首先通过空间编码器提取单历元内的卫星分布特征
    2. 时间增强: 引入LSTM-TFE时序编码器，学习信号的时间演变规律
    3. 交叉融合: 通过交叉注意力机制建立空间-时间的关联
    4. 稀疏压缩: 通过可学习激活函数Ω(z)获得更具判别性的稀疏表示
    5. 端到端分类: 稀疏特征直接输入分类器进行NLOS/LOS判断

══════════════════════════════════════════════════════════════════════════════
                              输入输出规格
══════════════════════════════════════════════════════════════════════════════

输入格式:
    - 单历元模式: (batch_size, input_dim) - 1D/2D张量
    - 窗口模式:   (batch_size, window_size, input_dim) - 3D张量

输出格式:
    - logits: (batch_size, num_classes) - 分类 logits
    - attention_weights (可选): (batch_size, query_len, key_len) - 注意力权重

══════════════════════════════════════════════════════════════════════════════
                              数学公式
══════════════════════════════════════════════════════════════════════════════

输入表示:
    x ∈ R^(B × T × F)
    
    其中:
    - B = batch size
    - T = 时间窗口大小 (window size)
    - F = 特征维度 (feature dimension)

空间特征提取:
    S = SpatialEncoder(x) = φ(x)
    
    其中:
    - S ∈ R^(B × T × d_s): 空间嵌入
    - φ(·): MLP非线性映射

时序特征提取 (LSTM-TFE):
    H = LSTM_TFE(S)
    
    其中:
    - H ∈ R^(B × d_h): 时序上下文向量
    - LSTM方程:
        f_t = σ(W_f · [h_{t-1}, s_t] + b_f)  # 遗忘门
        i_t = σ(W_i · [h_{t-1}, s_t] + b_i)  # 输入门
        C̃_t = tanh(W_c · [h_{t-1}, s_t] + b_c)  # 候选记忆
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t  # 单元状态
        o_t = σ(W_o · [h_{t-1}, s_t] + b_o)  # 输出门
        h_t = o_t ⊙ tanh(C_t)  # 隐藏状态

交叉注意力融合:
    A = CrossAttention(Q=S, K=H, V=H)
    
    其中:
    - A ∈ R^(B × T × d_a): 注意力增强特征
    - Attention(Q,K,V) = softmax(QK^T / √d)V

稀疏表示学习:
    Z = SP(A) = σ(W_s · A + b_s)
    
    其中:
    - Z ∈ R^(B × d_z): 稀疏表示
    - σ: 激活函数 (ReLU)
    - W_s ∈ R^(d_a × d_z): 稀疏投影矩阵

分类预测:
    ŷ = softmax(W_c · Z + b_c)
    
    其中:
    - ŷ ∈ R^(B × C): 预测概率
    - C = num_classes (2 for NLOS/LOS)

总损失函数:
    L = L_crossentropy(y, ŷ)
    
    其中:
    - L_crossentropy: 二元交叉熵损失（BCELoss）
    - 稀疏化通过可学习激活函数Ω(z)实现，无需额外L1损失

══════════════════════════════════════════════════════════════════════════════
                              关键参数配置建议
══════════════════════════════════════════════════════════════════════════════

基础参数:
    - input_dim: 你的GNSS特征数量
    - num_classes: 2 (NLOS/LOS二分类)

空间编码器 (AAM Module):
    - spatial_embed_dim: 64 (默认)
    - spatial_hidden_dims: [128, 64] (默认)
    - spatial_dropout: 0.1-0.2

时序编码器 (LSTM-TFE Module):
    - temporal_embed_dim: 64
    - temporal_num_layers: 1-2
    - temporal_dropout: 0.1-0.2
    - temporal_bidirectional: True (推荐)

交叉注意力:
    - cross_attn_embed_dim: 64
    - cross_attn_num_heads: 4
    - cross_attn_dropout: 0.1

稀疏表示 (SP(Z) Module):
    - sparse_embed_dim: 与交叉注意力/时间编码器输出维度一致（论文中为64）

分类器:
    - classifier_hidden_dims: [64, 32]
    - classifier_dropout: 0.3
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from pathlib import Path

# 添加项目根目录到路径，以便导入 utils 模块
_current_file = Path(__file__).resolve()
_module_dir = _current_file.parent      # stca/modules
_stca_dir = _module_dir.parent          # stca
_project_root = _stca_dir.parent        # DevLab (项目根目录)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

from .spatial_encoder import SpatialEncoder
from .temporal_encoder import TemporalEncoder
from .cross_attention import CrossAttention
from .sparse_representation import SparseRepresentation
from .constants import (
    SPATIAL_EMBED_DIM,
    SPATIAL_NUM_HEADS,
    SPATIAL_NUM_LAYERS,
    SPATIAL_D_FF,
    SPATIAL_DROPOUT,
    TEMPORAL_EMBED_DIM,
    TEMPORAL_NUM_LAYERS,
    TEMPORAL_DROPOUT,
    TEMPORAL_BIDIRECTIONAL,
    CROSS_ATTN_EMBED_DIM,
    CROSS_ATTN_NUM_HEADS,
    CROSS_ATTN_DROPOUT,
    CLASSIFIER_HIDDEN_DIMS,
    CLASSIFIER_DROPOUT,
)


class GNSSDualInputDataset(Dataset):
    """
    支持变长空间输入的 GNSS 数据集（类似 BaseModel）
    """
    def __init__(self, X_spatial_list, X_temporal, y):
        """
        Args:
            X_spatial_list: List of (N_i, 4) - 变长空间输入
            X_temporal: (N, window_size, 4) - 时间输入
            y: (N,) - 标签
        """
        self.X_spatial_list = X_spatial_list
        self.X_temporal = torch.FloatTensor(X_temporal)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X_temporal)

    def __getitem__(self, idx):
        return {
            'spatial': torch.FloatTensor(self.X_spatial_list[idx]),
            'temporal': self.X_temporal[idx],
            'label': self.y[idx]
        }


def dual_input_collate_fn(batch):
    """
    变长空间输入的 collate 函数（类似 BaseModel 的 combined_collate_fn）
    """
    temporal = torch.stack([b['temporal'] for b in batch], dim=0)
    labels = torch.stack([b['label'] for b in batch], dim=0)
    spatial_list = [b['spatial'] for b in batch]
    spatial_padded = pad_sequence(spatial_list, batch_first=True)

    return {
        'temporal': temporal,
        'spatial': spatial_padded,
        'labels': labels,
    }


class STCAModel(nn.Module):
    """
    GNSS NLOS 检测的 STCA 模型。
    
    时空交叉注意力模型，用于GNSS NLOS信号分类。
    支持窗口序列和非窗口（单历元）输入。
    
    模块组成:
        1. SpatialEncoder (AAM Module): 空间环境特征提取
        2. TemporalEncoder (LSTM-TFE Module): 时序信号建模
        3. CrossAttention: 空间-时间特征融合
        4. SparseRepresentation (SP(Z) Module): 稀疏特征压缩
        5. Classifier: 分类预测
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        # 空间编码器参数 (AAM Module) - 默认值来自 constants.py
        spatial_embed_dim: int = SPATIAL_EMBED_DIM,
        spatial_num_heads: int = SPATIAL_NUM_HEADS,
        spatial_num_layers: int = SPATIAL_NUM_LAYERS,
        spatial_d_ff: int = SPATIAL_D_FF,
        spatial_dropout: float = SPATIAL_DROPOUT,

        # 时间编码器参数 (LSTM-TFE Module) - 默认值来自 constants.py
        temporal_embed_dim: int = TEMPORAL_EMBED_DIM,
        temporal_num_layers: int = TEMPORAL_NUM_LAYERS,
        temporal_dropout: float = TEMPORAL_DROPOUT,
        temporal_bidirectional: bool = TEMPORAL_BIDIRECTIONAL,

        # 交叉注意力参数 - 默认值来自 constants.py
        cross_attn_embed_dim: int = CROSS_ATTN_EMBED_DIM,
        cross_attn_num_heads: int = CROSS_ATTN_NUM_HEADS,
        cross_attn_dropout: float = CROSS_ATTN_DROPOUT,

        # 分类器参数 - 默认值来自 constants.py
        classifier_hidden_dims: list = CLASSIFIER_HIDDEN_DIMS,
        classifier_dropout: float = CLASSIFIER_DROPOUT,

        # 消融实验控制参数
        use_cross_attention: bool = True,      # 是否使用交叉注意力模块
        use_sparse_representation: bool = True,  # 是否使用稀疏表示模块
    ):
        """
        参数:
            input_dim: 输入特征维度
            num_classes: 类别数量 (2 for NLOS/LOS)
            其他参数默认值来自 constants.py，可直接覆盖
            use_cross_attention: 是否使用交叉注意力模块 (消融实验用)
            use_sparse_representation: 是否使用稀疏表示模块 (消融实验用)
        """
        super(STCAModel, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_cross_attention = use_cross_attention
        self.use_sparse_representation = use_sparse_representation

        # 空间编码器 (AAM Module)
        self.spatial_encoder = SpatialEncoder(
            input_dim=input_dim,
            embed_dim=spatial_embed_dim,
            num_heads=spatial_num_heads,
            num_layers=spatial_num_layers,
            d_ff=spatial_d_ff,
            dropout_rate=spatial_dropout,
        )

        # 时间编码器 (LSTM-TFE Module)
        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            embed_dim=temporal_embed_dim,
            num_layers=temporal_num_layers,
            dropout_rate=temporal_dropout,
            bidirectional=temporal_bidirectional,
        )
        # 确定时间编码器输出维度
        if temporal_bidirectional:
            temporal_out_dim = temporal_embed_dim * 2
        else:
            temporal_out_dim = temporal_embed_dim

        # 交叉注意力（可选模块，论文 III-G）
        if use_cross_attention:
            self.cross_attention = CrossAttention(
                embed_dim=cross_attn_embed_dim,
                num_heads=cross_attn_num_heads,
                dropout_rate=cross_attn_dropout,
            )
            cross_attn_out_dim = self.cross_attention.embed_dim
        else:
            self.cross_attention = None
            # 无交叉注意力时，直接使用时间编码器输出
            cross_attn_out_dim = temporal_out_dim

        # 稀疏表示层（可选模块，论文 III-F）
        if use_sparse_representation:
            self.sparse_rep = SparseRepresentation(embed_dim=cross_attn_out_dim)
            classifier_input_dim = cross_attn_out_dim
        else:
            self.sparse_rep = None
            classifier_input_dim = cross_attn_out_dim
        # Classifier: 3层全连接层，符合论文IV.E节
        # 第1-2层: ReLU激活
        # 第3层(输出层): 1个神经元 + sigmoid激活
        layers = []
        prev_dim = classifier_input_dim

        for hidden_dim in classifier_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(classifier_dropout))
            prev_dim = hidden_dim

        # 输出层: 1个神经元 + sigmoid（论文要求）
        layers.append(nn.Linear(prev_dim, 1))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x_spatial=None, x_temporal=None, return_attention_weights=False, return_features=False):
        """
        前向传播

        参数:
            x_spatial: 空间编码器输入 (用于 AAM 模块)
                - 3D: (batch_size, window_size, input_dim)
            x_temporal: 时序编码器输入 (用于 LSTM-TFE 模块)
                - 3D: (batch_size, window_size, input_dim)
            return_attention_weights: 是否返回注意力权重
            return_features: 是否返回融合特征（用于 L1 正则化损失计算）

        Returns:
            logits: (batch_size, num_classes)
            attention_weights: (可选)
            fusion_features: (可选) 交叉注意力输出的融合特征，用于 L1 正则化
        """
        batch_size = x_spatial.size(0)
        
        # 空间编码 - 应用于 x_spatial
        spatial_emb = self.spatial_encoder(x_spatial)

        # 时间编码器 - 应用于 x_temporal
        temporal_emb = self.temporal_encoder(x_temporal)

        # 交叉注意力（可选模块，论文 III-G）
        if self.use_cross_attention:
            # Query: 时间特征 (batch, 1, temporal_out_dim)
            # Key/Value: 空间特征 (batch, num_sats, spatial_embed_dim)
            query = temporal_emb.unsqueeze(1)  # (batch, 1, temporal_out_dim)
            key = spatial_emb
            value = spatial_emb

            if return_attention_weights and self.cross_attention is not None:
                cross_attn_out, attn_weights = self.cross_attention(
                    query, key, value,
                    return_attention_weights=True
                )
            else:
                cross_attn_out = self.cross_attention(query, key, value) if self.cross_attention else None
                attn_weights = None

            # 稀疏表示（可选模块，论文 III-F）
            if self.use_sparse_representation and self.sparse_rep is not None:
                sparse_emb = self.sparse_rep(cross_attn_out.squeeze(1))
            else:
                sparse_emb = cross_attn_out.squeeze(1)
        else:
            # 无交叉注意力：直接使用时间编码器输出
            attn_weights = None
            cross_attn_out = None
            
            if self.use_sparse_representation and self.sparse_rep is not None:
                sparse_emb = self.sparse_rep(temporal_emb)
            else:
                sparse_emb = temporal_emb

        # 分类预测
        logits = self.classifier(sparse_emb)

        # 返回结果
        if return_attention_weights and attn_weights is not None:
            if return_features:
                return logits, attn_weights, cross_attn_out
            return logits, attn_weights
        elif return_features:
            return logits, cross_attn_out
        else:
            return logits

    def fit(self, X_train_spatial, y_train, X_val_spatial=None, y_val=None,
            epochs=50, batch_size=16, lr=0.001, device='cpu',
            verbose=True, progress_callback=None,
            X_train_temporal=None, X_val_temporal=None,
            use_scheduler=True, scheduler_factor=0.5, scheduler_patience=5, scheduler_min_lr=1e-6,
            weight_decay=1e-4):
        """
        训练模型方法。

        参数说明:
            X_train_spatial: 训练空间特征，2 维数组 (samples, max_satellites, features) 训练特征，2维或3维数组
            y_train: 训练标签
            X_val_spatial: 验证集特征（可选）
            y_val: 验证集标签（可选）
            epochs: 训练轮数
            batch_size: 批量大小
            lr: 学习率
            device: 用于训练的设备（如 'cpu' 或 'cuda'）
            verbose: 是否输出训练进度
            progress_callback: 进度回调函数，可选，格式为(current_epoch, total_epochs)
            X_train_temporal: 训练时间特征，3 维数组 (samples, window_size, features)， 训练用的时间输入，3维数组（batch, window, features），可选
            X_val_temporal: 验证时间特征，3 维数组 (samples, window_size, features)， 验证用的时间输入，3维数组（batch, window, features），可选

        返回:
            训练过程历史字典
        """
   
        # 处理双输入：空间 (2D) + 时间 (3D)
        use_dual_input = X_train_temporal is not None
        self.to(device)
        self.train()

        # 根据输入类型准备数据
        if use_dual_input:
            # 使用分离的空间 (2D) 和时间 (3D) 输入
            # X_train_spatial 和 X_train_temporal 已经是参数传入的值
            if X_val_spatial is not None and X_val_temporal is not None:
                # X_val_spatial 和 X_val_temporal 已经是参数传入的值
                pass
            else:
                X_val_spatial = None
                X_val_temporal = None
        else:
            # 自动检测：如果 X_train_spatial 是 3D，用于时间模型
            X_train_np = np.array(X_train_spatial) if not isinstance(X_train_spatial, np.ndarray) else X_train_spatial

            if X_train_np.ndim == 3:
                # 3D 数据：提取空间（最后时间步）并使用全部作为时间
                X_train_spatial = X_train_np[:, -1, :]
                X_train_temporal = X_train_np

                if X_val_spatial is not None:
                    X_val_np = np.array(X_val_spatial) if not isinstance(X_val_spatial, np.ndarray) else X_val_spatial
                    X_val_spatial = X_val_np[:, -1, :]
                    X_val_temporal = X_val_np
                else:
                    X_val_spatial = None
                    X_val_temporal = None
            else:
                # 2D 数据：使用单一输入
                X_train_temporal = None
                X_val_temporal = None

        # 创建数据加载器（如需双输入）
        # 确保标签转换为 float 类型（BCELoss 需要）
        y_train_arr = np.array(y_train, dtype=np.float32)

        # 检测是否为变长空间输入（List 格式）
        is_variable_length = isinstance(X_train_spatial, list)

        if is_variable_length:
            # 变长空间输入：使用 GNSSDualInputDataset + pad_sequence collate_fn
            logger.info("Using variable-length spatial input with padding collate")
            train_dataset = GNSSDualInputDataset(X_train_spatial, X_train_temporal, y_train_arr)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                     collate_fn=dual_input_collate_fn)
        elif use_dual_input or X_train_temporal is not None:
            # 固定长度空间输入：使用 TensorDataset
            train_spatial = torch.FloatTensor(X_train_spatial)
            train_temporal = torch.FloatTensor(X_train_temporal)
            train_labels = torch.FloatTensor(y_train_arr)
            train_dataset = TensorDataset(train_spatial, train_temporal, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        else:
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_spatial),
                torch.FloatTensor(y_train_arr)
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


        # 优化器：添加 L2 正则化（权重衰减），防止过拟合
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)

        # 学习率调度器：ReduceLROnPlateau（验证集 Loss 不下降时降低学习率）
        scheduler = None
        if use_scheduler and X_val_spatial is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=scheduler_factor,
                patience=scheduler_patience, min_lr=scheduler_min_lr
            )

        # Loss function: 二元交叉熵损失（论文 IV.E 节）
        # 使用 BCEWithLogitsLoss 替代 BCELoss，数值更稳定
        # 添加类别权重处理类别不平衡问题（类似 BaseModel 的做法）
        count_label1 = np.sum(y_train_arr == 1)
        count_label0 = np.sum(y_train_arr == 0)
        pos_weight = count_label0 / max(1, count_label1)
        logger.info(f"类别分布：NLOS={count_label0}, LOS={count_label1}, pos_weight={pos_weight:.4f}")

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

        history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
        }
        

        # 检查是否需要双输入

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            all_train_preds = []
            all_train_targets = []

            for batch_data in train_loader:
                if use_dual_input or is_variable_length:
                    # 双输入或变长输入：batch_data 是 dict (来自 dual_input_collate_fn)
                    # {'spatial': padded_batch, 'temporal': batch, 'labels': batch}
                    batch_x_spatial = batch_data['spatial']
                    batch_x_temporal = batch_data['temporal']
                    batch_y = batch_data['labels']
                    batch_x_spatial = batch_x_spatial.to(device)
                    batch_x_temporal = batch_x_temporal.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = self.forward(x_spatial=batch_x_spatial, x_temporal=batch_x_temporal)
                else:
                    # 单输入
                    batch_x, batch_y = batch_data
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = self.forward(batch_x)

                # 论文 III-F：稀疏化由 Ω(z) 激活实现，无需额外 L1 损失
                # BCEWithLogitsLoss 接收 logits（未通过 sigmoid），内部处理数值稳定性
                loss = criterion(outputs.squeeze(-1), batch_y)

                loss.backward()
                # 梯度裁剪：防止梯度爆炸，稳定训练（类似 BaseModel 的正则化效果）
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item() * batch_y.size(0)
                # sigmoid输出概率值，使用阈值0.5判断类别
                preds = (torch.sigmoid(outputs.squeeze(-1)) >= 0.5).float()
                epoch_total += batch_y.size(0)
                epoch_correct += (preds == batch_y).sum().item()

                # 收集 predictions 和 targets 用于计算 F1 分数
                all_train_preds.extend(preds.cpu().numpy())
                all_train_targets.extend(batch_y.cpu().numpy())

            epoch_loss /= epoch_total
            epoch_acc = epoch_correct / epoch_total

            # F1 分数（需要 sklearn）
            from sklearn.metrics import f1_score
            train_f1 = 100. * f1_score(all_train_targets, all_train_preds, average='binary', zero_division=0)

            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            history['train_f1'].append(train_f1)

            # 验证
            if X_val_spatial is not None and y_val is not None:
                self.eval()

                # 创建验证数据集，基于输入类型
                # 确保标签转换为 float 类型
                y_val_arr = np.array(y_val, dtype=np.float32)
                
                # 检测是否为变长空间输入（List 格式）
                is_variable_length = isinstance(X_val_spatial, list)
                
                if is_variable_length:
                    # 变长空间输入：使用 GNSSDualInputDataset + pad_sequence collate_fn
                    val_dataset = GNSSDualInputDataset(X_val_spatial, X_val_temporal, y_val_arr)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                           collate_fn=dual_input_collate_fn)
                elif use_dual_input and X_val_temporal is not None:
                    val_spatial = torch.FloatTensor(X_val_spatial)
                    val_temporal = torch.FloatTensor(X_val_temporal)
                    val_labels = torch.FloatTensor(y_val_arr)
                    val_dataset = TensorDataset(val_spatial, val_temporal, val_labels)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                else:
                    val_dataset = TensorDataset(
                        torch.FloatTensor(X_val_spatial),
                        torch.FloatTensor(y_val_arr)
                    )
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                # 验证循环
                val_loss = 0
                val_correct = 0
                val_total = 0
                all_val_preds = []
                all_val_targets = []

                with torch.no_grad():
                    for val_data in val_loader:
                        if use_dual_input or is_variable_length:
                            # 双输入或变长输入：batch_data 是 dict (来自 dual_input_collate_fn)
                            batch_x_spatial = val_data['spatial']
                            batch_x_temporal = val_data['temporal']
                            batch_y = val_data['labels']
                            batch_x_spatial = batch_x_spatial.to(device)
                            batch_x_temporal = batch_x_temporal.to(device)
                            batch_y = batch_y.to(device)

                            outputs = self.forward(x_spatial=batch_x_spatial, x_temporal=batch_x_temporal)
                        else:
                            batch_x, batch_y = val_data
                            batch_x = batch_x.to(device)
                            batch_y = batch_y.to(device)

                            outputs = self.forward(batch_x)

                        # 使用 clamp 确保数值稳定性，避免 log(0)（与训练时一致）
                        # BCEWithLogitsLoss 接收 logits，无需 clamp
                        loss = criterion(outputs.squeeze(-1), batch_y)

                        val_loss += loss.item() * batch_y.size(0)
                        # sigmoid输出概率值，使用阈值0.5判断类别
                        preds = (torch.sigmoid(outputs.squeeze(-1)) >= 0.5).float()
                        val_total += batch_y.size(0)
                        val_correct += (preds == batch_y).sum().item()

                        # 收集 predictions 和 targets 用于计算 F1 分数
                        all_val_preds.extend(preds.cpu().numpy())
                        all_val_targets.extend(batch_y.cpu().numpy())

                val_loss /= val_total
                val_acc = val_correct / val_total
                val_f1 = 100. * f1_score(all_val_targets, all_val_preds, average='binary', zero_division=0)

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['val_f1'].append(val_f1)

                # 学习率调度器 step
                if scheduler is not None:
                    scheduler.step(val_loss)

                if verbose:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Train F1: {train_f1:.2f}%, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.2f}%"
                    )
            else:
                if verbose:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

            # 调用进度回调函数（如提供）
            if progress_callback:
                progress_callback(epoch + 1, epochs)
        
        return history

    def evaluate(self, X_test, y_test, device='cpu', X_test_3d=None):
        """Evaluate model on test set.

        参数:
            X_test: Test features (2D/3D tensor, or list for variable-length spatial)
            y_test: Test labels
            device: Device to use
            X_test_3d: Optional 3D test data for temporal models
        """
        self.to(device)
        self.eval()

        # 确定是否有双输入
        if X_test_3d is not None:
            # 检测是否为变长空间输入（List 格式）
            is_variable_length = isinstance(X_test, list)
            
            y_test_arr = np.array(y_test, dtype=np.float32)
            
            if is_variable_length:
                # 变长空间输入：使用 GNSSDualInputDataset + collate_fn
                test_dataset = GNSSDualInputDataset(X_test, X_test_3d, y_test_arr)
                test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                                        collate_fn=dual_input_collate_fn)
            else:
                # 固定长度空间输入：使用 TensorDataset
                X_test_spatial = X_test
                X_test_temporal = X_test_3d
                test_dataset = TensorDataset(
                    torch.FloatTensor(X_test_spatial),
                    torch.FloatTensor(X_test_temporal),
                    torch.FloatTensor(y_test_arr)
                )
                test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        all_preds = []
        all_probs = []

        dual_input = X_test_3d is not None or (X_test_np.ndim == 3 if 'X_test_np' in dir() else False)

        with torch.no_grad():
            for batch_data in test_loader:
                if isinstance(batch_data, dict):
                    # 变长输入：batch_data 是 dict (来自 dual_input_collate_fn)
                    batch_x_spatial = batch_data['spatial']
                    batch_x_temporal = batch_data['temporal']
                    batch_x_spatial = batch_x_spatial.to(device)
                    batch_x_temporal = batch_x_temporal.to(device)
                    outputs = self.forward(x_spatial=batch_x_spatial, x_temporal=batch_x_temporal)
                elif len(batch_data) == 3:
                    # 双输入（固定长度）
                    batch_x_spatial, batch_x_temporal, _ = batch_data
                    batch_x_spatial = batch_x_spatial.to(device)
                    batch_x_temporal = batch_x_temporal.to(device)
                    outputs = self.forward(x_spatial=batch_x_spatial, x_temporal=batch_x_temporal)
                else:
                    # 单输入
                    batch_x, _ = batch_data
                    batch_x = batch_x.to(device)
                    outputs = self.forward(batch_x)
                # sigmoid输出概率值，形状为(batch, 1)
                # 阈值0.5判断类别
                probs = torch.sigmoid(outputs.squeeze(-1))  # (batch, 1) -> (batch,)
                preds = (probs >= 0.5).long()  # >=0.5为LOS(1)，<0.5为NLOS(0)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

        # 计算 ROC AUC
        # sigmoid输出单个概率值（LOS的概率），直接用于roc_curve
        if all_probs is not None and len(all_probs.shape) == 1:
            fpr, tpr, _ = roc_curve(y_test, all_probs)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = 0.0

        return {
            'accuracy': accuracy_score(y_test, all_preds),
            'precision': precision_score(y_test, all_preds, average='binary', zero_division=0),
            'recall': recall_score(y_test, all_preds, average='binary', zero_division=0),
            'f1_score': f1_score(y_test, all_preds, average='binary', zero_division=0),
            'roc_auc': roc_auc,
            'y_pred': all_preds,
            'y_proba': all_probs,
        }

