# stca_model.py
"""
Spatiotemporal Cross-Attention (STCA) Model with Sparse Representation
时空交叉注意力模型 (带稀疏表示) - PyTorch Version
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
                              模型架构详解
══════════════════════════════════════════════════════════════════════════════

完整的数据流 (当所有模块启用时):

    ┌──────────────────────────────────────────────────────────────────────┐
    │                         原始GNSS输入                                  │
    │              (batch, window_size, input_dim)                        │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │         SpatialEncoder (空间环境特征提取模块 / AAM Module)             │
    │   输入: 原始GNSS特征  →  输出: 空间嵌入 (batch, window, d)            │
    │   对每个时间步分别编码，提取单历元空间特征                               │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │    TemporalEncoder (时序信号建模模块 / LSTM-TFE Module)              │
    │     输入: 空间嵌入序列  →  输出: 时间上下文 (batch, d)                │
    │     LSTM网络学习信号的时间演变规律                                    │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │      CrossAttention (时空交叉注意力融合模块 / Cross-Attention)        │
    │  输入: query=空间特征, key/value=时间特征                            │
    │  输出: 空间-时间融合特征 (batch, d)                                   │
    │  建立空间与时间特征的依赖关系                                         │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │     SparseRepresentation (稀疏表示学习模块 / SP(Z) Module)           │
    │      输入: 融合特征  →  输出: 稀疏嵌入 (batch, d_sparse)             │
    │      L1正则化压缩特征，增强可解释性                                   │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                      Classifier (分类器)                              │
    │       输入: 稀疏嵌入  →  输出: NLOS/LOS概率 (batch, 2)               │
    │       MLP分类器，输出softmax概率分布                                  │
    └──────────────────────────────────────────────────────────────────────┘

══════════════════════════════════════════════════════════════════════════════
                              四种模型配置模式
══════════════════════════════════════════════════════════════════════════════

模式1: 纯空间模型 (use_temporal=False, use_cross_attention=False)
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 输入 → SpatialEncoder → MeanPool → SparseRep → Classifier → 输出  │
    └─────────────────────────────────────────────────────────────────────┘
    适用场景: 单历元预测，计算资源有限

模式2: 时序模型 (use_temporal=True, use_cross_attention=False)
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 输入 → SpatialEncoder → TemporalEncoder → SparseRep → Classifier    │
    └─────────────────────────────────────────────────────────────────────┘
    适用场景: 窗口序列预测，关注时间演变

模式3: 交叉注意力模型 (use_temporal=False, use_cross_attention=True)
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 输入 → SpatialEncoder → CrossAttention → SparseRep → Classifier    │
    └─────────────────────────────────────────────────────────────────────┘
    适用场景: 空间自关联学习

模式4: 完整STCA模型 (use_temporal=True, use_cross_attention=True)
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 输入 → SpatialEncoder → TemporalEncoder → CrossAttention            │
    │                              → SparseRep → Classifier               │
    └─────────────────────────────────────────────────────────────────────┘
    适用场景: 充分利用时空信息，效果最佳

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from spatial_encoder import SpatialEncoder
from temporal_encoder import TemporalEncoder
from cross_attention import CrossAttention
from sparse_representation import SparseRepresentation


class STCAModel(nn.Module):
    """
    STCA Model for GNSS NLOS detection.
    
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
        # Spatial encoder params (AAM Module)
        spatial_embed_dim: int = 64,
        spatial_num_heads: int = 1,
        spatial_num_layers: int = 1,
        spatial_d_ff: int = None,
        spatial_hidden_dims: list = None,
        spatial_dropout: float = 0.1,
        # Temporal encoder params (LSTM-TFE Module)
        # 始终启用时序编码器
        temporal_embed_dim: int = 64,
        temporal_num_layers: int = 1,
        temporal_dropout: float = 0.1,
        temporal_bidirectional: bool = False,
        # Cross-attention params
        use_cross_attention: bool = False,
        cross_attn_embed_dim: int = 64,
        cross_attn_num_heads: int = 1,
        cross_attn_dropout: float = 0.1,
        # Sparse representation params (SP(Z) Module) — 论文 III-F：仅 embed_dim，无额外参数
        # Classification params
        classifier_hidden_dims: list = None,
        classifier_dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: 输入特征维度
            num_classes: 类别数量 (2 for NLOS/LOS)
            use_cross_attention: 是否使用交叉注意力模块
            其他参数映射到各个子模块
        """
        super(STCAModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_cross_attention = use_cross_attention

        # Spatial Encoder (AAM Module)
        # Spatial Encoder (AAM Module)
        # Now using Attention Aggregation Module instead of MLP
        self.spatial_encoder = SpatialEncoder(
            input_dim=input_dim,
            embed_dim=spatial_embed_dim,
            num_heads=spatial_num_heads,
            num_layers=spatial_num_layers,
            d_ff=spatial_d_ff,
            dropout_rate=spatial_dropout,
        )

        # Temporal Encoder (LSTM-TFE Module)
        # 始终启用时序编码器 (根据实验要求)
        # Temporal encoder receives raw temporal features directly (without spatial encoder)
        # Input dimension should be the original feature dimension (input_dim)
        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,  # Raw feature dimension, not spatial_embed_dim
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

        # Cross-Attention
        # 注意: Query 来自时间特征 (temporal_out_dim), Key/Value 来自空间特征 (spatial_embed_dim)
        self.cross_attention = None
        cross_attn_out_dim = None

        if use_cross_attention:
            # 确定 query 维度 (来自 temporal) 和 kv 维度 (来自 spatial)
            query_input_dim = temporal_out_dim if temporal_out_dim else spatial_embed_dim
            kv_input_dim = spatial_embed_dim

            self.cross_attention = CrossAttention(
                embed_dim=cross_attn_embed_dim,
                num_heads=cross_attn_num_heads,
                dropout_rate=cross_attn_dropout,
                query_dim=query_input_dim,
                kv_dim=kv_input_dim,
            )
            cross_attn_out_dim = cross_attn_embed_dim

        # 稀疏表示直接作用在融合特征上，仅需 embed_dim（论文 III-F/III-G）
        # 有交叉注意力时：输入为交叉注意力输出 dim；否则为时序编码器输出 dim
        sparse_embed_dim = cross_attn_out_dim if use_cross_attention else temporal_out_dim
        self.sparse_rep = SparseRepresentation(embed_dim=sparse_embed_dim)

        # Classifier: 3层全连接层，符合论文IV.E节
        # 第1-2层: ReLU激活
        # 第3层(输出层): 1个神经元 + sigmoid激活
        classifier_hidden_dims = classifier_hidden_dims or [64, 32]
        layers = []
        prev_dim = sparse_embed_dim

        for i, hidden_dim in enumerate(classifier_hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(classifier_dropout))
            prev_dim = hidden_dim

        # 输出层: 1个神经元 + sigmoid（论文要求）
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*layers)

    def forward(self, x_spatial=None, x_temporal=None, return_attention_weights=False, return_features=False):
        """
        前向传播

        Args:
            x_spatial: 空间编码器输入 (用于 AAM 模块)
                - 2D: (batch_size, input_dim) 或 3D: (batch_size, window_size, input_dim)
                - 如果为 None，则从 x_temporal 提取
            x_temporal: 时序编码器输入 (用于 LSTM-TFE 模块)
                - 必须为 3D: (batch_size, window_size, input_dim)
                - 如果 use_temporal=True 则必须提供
            return_attention_weights: 是否返回注意力权重
            return_features: 是否返回融合特征（用于 L1 正则化损失计算）

        Returns:
            logits: (batch_size, num_classes)
            attention_weights: (可选)
            fusion_features: (可选) 交叉注意力输出的融合特征，用于 L1 正则化
        """
        # 确定使用的输入
        if x_temporal is not None and x_spatial is None:
            # 如果只提供 3D 数据，用它同时作为空间和时序输入
            x_spatial = x_temporal
        elif x_spatial is not None and x_temporal is None:
            # 如果只提供 2D 数据或 3D 数据但不使用时序
            x_temporal = x_spatial
        
        # 调整输入维度
        if x_spatial.dim() == 2:
            # (batch, features) -> (batch, 1, features)
            x_spatial = x_spatial.unsqueeze(1)
        
        if x_temporal.dim() == 2:
            # 2D input cannot be used for temporal encoding
            # Use spatial input for temporal as well (window=1)
            x_temporal = x_spatial
        
        batch_size = x_spatial.size(0)
        
        # 空间编码 (Spatial Encoding) - 应用于 x_spatial
        # x_spatial: (batch, window, input_dim)
        # spatial_emb: (batch, window, spatial_embed_dim)
        spatial_emb = self.spatial_encoder(x_spatial)

        # 时序编码 (Temporal Encoding) - 应用于 x_temporal
        # 始终启用时序编码器
        # x_temporal should be 3D for temporal encoding
        if x_temporal.dim() == 2:
            x_temporal = x_temporal.unsqueeze(1)

        # Temporal encoder directly processes the temporal sequence
        # WITHOUT passing through spatial encoder (as per paper architecture)
        # temporal_emb: (batch, temporal_out_dim)
        temporal_emb = self.temporal_encoder(x_temporal)

        # 交叉注意力 (Cross-Attention)
        # 根据论文: Temporal features query Spatial features
        # Q = H_temporal * W_q (时间特征作为Query)
        # K = H_spatial * W_k (空间特征作为Key)
        # V = H_spatial * W_v (空间特征作为Value)
        # Attention(Q,K,V) = softmax(QK^T / √d_k) V
        cross_attn_out = None
        attn_weights = None

        if self.use_cross_attention:
            # 论文 III-G 公式(17)：单时间特征向量作 Query，空间卫星序列作 Key/Value
            # Query: (batch, 1, query_dim)，不扩展为与空间序列同长
            if temporal_emb is not None:
                query = temporal_emb.unsqueeze(1)  # (batch, 1, temporal_out_dim)
            else:
                query = spatial_emb[:, :1, :]  # 退化为取空间首帧 (batch, 1, spatial_embed_dim)

            key = spatial_emb   # (batch, window, spatial_embed_dim)
            value = spatial_emb  # (batch, window, spatial_embed_dim)

            if return_attention_weights:
                cross_attn_out, attn_weights = self.cross_attention(
                    query, key, value,
                    return_attention_weights=True
                )
                # cross_attn_out 已是 (batch, 1, embed_dim)，稀疏层会 squeeze 为 (batch, embed_dim)
            else:
                cross_attn_out = self.cross_attention(query, key, value)

        # 稀疏表示 (Sparse Representation)
        if self.use_cross_attention:
            sparse_input = cross_attn_out
        else:
            # 使用时序特征的均值池化
            sparse_input = temporal_emb

        # 应用稀疏表示层获取潜在嵌入
        sparse_emb = self.sparse_rep(sparse_input)

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

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=50, batch_size=256, lr=0.001, device='cpu',
            verbose=True, progress_callback=None,
            x_temporal_train=None, x_temporal_val=None):
        """
        Train the model.

        Args:
            X_train: Training features (2D or 3D)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            device: Device to train on
            verbose: Print training progress
            progress_callback: Optional callback function(current_epoch, total_epochs)
            x_temporal_train: Optional 3D temporal input for training (batch, window, features)
            x_temporal_val: Optional 3D temporal input for validation (batch, window, features)

        Returns:
            Training history dictionary
        """
        # Handle dual input: spatial (2D) + temporal (3D)
        # If x_temporal is provided, use it; otherwise infer from X_train shape
        use_dual_input = x_temporal_train is not None

        self.to(device)
        self.train()

        # Prepare data based on input type
        if use_dual_input:
            # Use separate spatial (2D) and temporal (3D) inputs
            X_train_spatial = X_train
            X_train_temporal = x_temporal_train

            if X_val is not None and x_temporal_val is not None:
                X_val_spatial = X_val
                X_val_temporal = x_temporal_val
            else:
                X_val_spatial = None
                X_val_temporal = None
        else:
            # Auto-detect: if X_train is 3D, use it for temporal model
            X_train_np = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train

            if X_train_np.ndim == 3:
                # 3D data: extract spatial (last timestep) and use full as temporal
                X_train_spatial = X_train_np[:, -1, :]
                X_train_temporal = X_train_np

                if X_val is not None:
                    X_val_np = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val
                    X_val_spatial = X_val_np[:, -1, :]
                    X_val_temporal = X_val_np
                else:
                    X_val_spatial = None
                    X_val_temporal = None
            else:
                # 2D data: use single input
                X_train_spatial = X_train
                X_train_temporal = None
                X_val_spatial = X_val
                X_val_temporal = None

        # Create data loaders with dual inputs if needed
        # 确保标签转换为float类型（BCELoss需要）
        y_train_arr = np.array(y_train, dtype=np.float32)
        if use_dual_input or X_train_temporal is not None:
            # Create datasets with both spatial and temporal data
            train_spatial = torch.FloatTensor(X_train_spatial)
            train_temporal = torch.FloatTensor(X_train_temporal)
            # BCELoss需要float类型标签
            train_labels = torch.FloatTensor(y_train_arr)
            train_dataset = TensorDataset(train_spatial, train_temporal, train_labels)
        else:
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_spatial),
                torch.FloatTensor(y_train_arr)  # BCE需要float类型
            )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.98))
        
        # Loss function: 二元交叉熵损失（论文IV.E节）
        # 使用BCELoss因为模型输出层已经包含sigmoid
        criterion = nn.BCELoss()
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        # Progress tracking
        import threading
        progress_lock = threading.Lock()
        progress_interval = max(1, epochs // 10)  # Update every 10% of epochs

        # Check if we need dual input
        dual_input = X_train_temporal is not None

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0

            for batch_data in train_loader:
                if dual_input:
                    # Dual input: (spatial, temporal, labels)
                    batch_x_spatial, batch_x_temporal, batch_y = batch_data
                    batch_x_spatial = batch_x_spatial.to(device)
                    batch_x_temporal = batch_x_temporal.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = self.forward(x_spatial=batch_x_spatial, x_temporal=batch_x_temporal)
                else:
                    # Single input
                    batch_x, batch_y = batch_data
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = self.forward(batch_x)

                # 论文 III-F：稀疏化由 Ω(z) 激活实现，无需额外 L1 损失
                loss = criterion(outputs.squeeze(-1), batch_y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_y.size(0)
                # sigmoid输出概率值，使用阈值0.5判断类别
                preds = (outputs.squeeze(-1) >= 0.5).float()
                epoch_total += batch_y.size(0)
                epoch_correct += (preds == batch_y).sum().item()

            epoch_loss /= epoch_total
            epoch_acc = epoch_correct / epoch_total

            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)

            # Validation
            if X_val_spatial is not None and y_val is not None:
                self.eval()

                # 创建验证数据集，基于输入类型
                # 确保标签转换为float类型
                y_val_arr = np.array(y_val, dtype=np.float32)
                if dual_input and X_val_temporal is not None:
                    val_spatial = torch.FloatTensor(X_val_spatial)
                    val_temporal = torch.FloatTensor(X_val_temporal)
                    val_labels = torch.FloatTensor(y_val_arr)
                    val_dataset = TensorDataset(val_spatial, val_temporal, val_labels)
                else:
                    val_dataset = TensorDataset(
                        torch.FloatTensor(X_val_spatial) if X_val_spatial is not None else torch.FloatTensor(X_val),
                        torch.FloatTensor(y_val_arr)
                    )

                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for val_data in val_loader:
                        if dual_input:
                            batch_x_spatial, batch_x_temporal, batch_y = val_data
                            batch_x_spatial = batch_x_spatial.to(device)
                            batch_x_temporal = batch_x_temporal.to(device)
                            batch_y = batch_y.to(device)

                            outputs = self.forward(x_spatial=batch_x_spatial, x_temporal=batch_x_temporal)
                        else:
                            batch_x, batch_y = val_data
                            batch_x = batch_x.to(device)
                            batch_y = batch_y.to(device)

                            outputs = self.forward(batch_x)

                        loss = criterion(outputs.squeeze(-1), batch_y)

                        val_loss += loss.item() * batch_y.size(0)
                        # sigmoid输出概率值，使用阈值0.5判断类别
                        preds = (outputs.squeeze(-1) >= 0.5).float()
                        val_total += batch_y.size(0)
                        val_correct += (preds == batch_y).sum().item()
                
                val_loss /= val_total
                val_acc = val_correct / val_total
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, "
                          f"Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(epoch + 1, epochs)
        
        return history

    def evaluate(self, X_test, y_test, device='cpu', X_test_3d=None):
        """Evaluate model on test set.

        Args:
            X_test: Test features (2D or 3D)
            y_test: Test labels
            device: Device to use
            X_test_3d: Optional 3D test data for temporal models
        """
        self.to(device)
        self.eval()

        # Determine if we have dual input
        if X_test_3d is not None:
            # Use dual input (spatial 2D + temporal 3D)
            X_test_spatial = X_test
            X_test_temporal = X_test_3d

            y_test_arr = np.array(y_test, dtype=np.float32)
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test_spatial),
                torch.FloatTensor(X_test_temporal),
                torch.FloatTensor(y_test_arr)  # BCELoss 需要 FloatTensor（论文 IV.E）
            )
        else:
            # Auto-detect: if X_test is 3D, use it
            X_test_np = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
            y_test_arr = np.array(y_test, dtype=np.float32)
            if X_test_np.ndim == 3:
                X_test_spatial = X_test_np[:, -1, :]
                X_test_temporal = X_test_np

                test_dataset = TensorDataset(
                    torch.FloatTensor(X_test_spatial),
                    torch.FloatTensor(X_test_temporal),
                    torch.FloatTensor(y_test_arr)  # BCELoss 需要 FloatTensor
                )
            else:
                # Single input
                test_dataset = TensorDataset(
                    torch.FloatTensor(X_test),
                    torch.FloatTensor(y_test_arr)
                )

        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        all_preds = []
        all_probs = []

        dual_input = X_test_3d is not None or (X_test_np.ndim == 3 if 'X_test_np' in dir() else False)

        with torch.no_grad():
            for batch_data in test_loader:
                if len(batch_data) == 3:
                    # Dual input
                    batch_x_spatial, batch_x_temporal, _ = batch_data
                    batch_x_spatial = batch_x_spatial.to(device)
                    batch_x_temporal = batch_x_temporal.to(device)
                    outputs = self.forward(x_spatial=batch_x_spatial, x_temporal=batch_x_temporal)
                else:
                    # Single input
                    batch_x, _ = batch_data
                    batch_x = batch_x.to(device)
                    outputs = self.forward(batch_x)

                # sigmoid输出概率值，形状为(batch, 1)
                # 阈值0.5判断类别
                probs = outputs.squeeze(-1)  # (batch, 1) -> (batch,)
                preds = (probs >= 0.5).long()  # >=0.5为LOS(1)，<0.5为NLOS(0)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

        # Calculate ROC AUC
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
    
    def get_config(self):
        """返回模型配置"""
        config = {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "use_cross_attention": self.use_cross_attention,
        }
        
        if self.spatial_encoder:
            config.update(self.spatial_encoder.get_config())
        
        if self.temporal_encoder:
            config.update(self.temporal_encoder.get_config())
        
        if self.cross_attention:
            config.update(self.cross_attention.get_config())
        
        if self.sparse_rep:
            config.update(self.sparse_rep.get_config())
        
        return config


# 单元测试
if __name__ == "__main__":
    # 测试 STCAModel（__init__ 无 use_temporal 参数，时序编码器始终启用）
    model = STCAModel(
        input_dim=9,
        use_cross_attention=True,
        temporal_bidirectional=True
    )
    
    # 测试单历元输入
    x_single = torch.randn(32, 9)  # batch=32, features=9
    out_single = model(x_single)
    print(f"Single Epoch Input: {x_single.shape} -> Output: {out_single.shape}")
    
    # 测试窗口输入
    x_window = torch.randn(32, 10, 9)  # batch=32, window=10, features=9
    out_window = model(x_window)
    print(f"Window Input: {x_window.shape} -> Output: {out_window.shape}")
    
    # 测试带注意力权重
    out, attn = model(x_window, return_attention_weights=True)
    print(f"Attention weights shape: {attn.shape}")
    
    # 打印模型结构
    print("\nModel Architecture:")
    print(model)
    
    # 测试参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
