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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
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
    SPARSE_EMBED_DIM,
    CLASSIFIER_HIDDEN_DIMS,
    CLASSIFIER_DROPOUT,
)


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

        # 稀疏表示参数 - 默认值来自 constants.py
        sparse_embed_dim: int = SPARSE_EMBED_DIM,

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
                query_input_dim=temporal_out_dim,  # 时间编码器输出维度
                kv_input_dim=spatial_embed_dim,    # 空间编码器输出维度
            )
            # 交叉注意力输出 + 时间特征拼接，保留完整信息
            fusion_out_dim = cross_attn_embed_dim + temporal_out_dim
        else:
            self.cross_attention = None
            # Concat 基线模型：拼接时间 + 空间特征
            # 空间特征经过平均池化后维度为 spatial_embed_dim
            fusion_out_dim = temporal_out_dim + spatial_embed_dim

        # 稀疏表示层（可选模块，论文 III-F）
        if use_sparse_representation:
            self.sparse_rep = SparseRepresentation(
                embed_dim=fusion_out_dim,
                sparse_embed_dim=sparse_embed_dim,
            )
            classifier_input_dim = sparse_embed_dim
        else:
            self.sparse_rep = None
            classifier_input_dim = fusion_out_dim
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
        layers.append(nn.Sigmoid())

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

            # 拼接时间特征和交叉注意力输出，保留完整信息
            # cross_attn_out: (batch, 1, embed_dim), temporal_emb: (batch, temporal_out_dim)
            fused_features = torch.cat([cross_attn_out.squeeze(1), temporal_emb], dim=-1)

            # 稀疏表示（可选模块，论文 III-F）
            if self.use_sparse_representation and self.sparse_rep is not None:
                sparse_emb = self.sparse_rep(fused_features)
            else:
                sparse_emb = fused_features
        else:
            # 无交叉注意力：Concat 基线模型（论文对比实验）
            # 将空间特征和时间特征直接拼接
            attn_weights = None
            cross_attn_out = None

            # 空间特征聚合：对 num_sats 维度取平均池化，得到 (batch, spatial_embed_dim)
            spatial_pooled = spatial_emb.mean(dim=1)  # (batch, 64)

            # Concat 拼接：[h_t, h_s] -> (batch, temporal_out_dim + spatial_embed_dim)
            fused_features = torch.cat([temporal_emb, spatial_pooled], dim=-1)

            if self.use_sparse_representation and self.sparse_rep is not None:
                sparse_emb = self.sparse_rep(fused_features)
            else:
                # 基线情况：直接输出拼接特征，不做稀疏变换
                sparse_emb = fused_features

            # 用于 return_features 的融合特征
            fusion_features = fused_features

        # 分类预测
        logits = self.classifier(sparse_emb)

        # 返回结果
        if return_attention_weights and attn_weights is not None:
            if return_features:
                return logits, attn_weights, fusion_features
            return logits, attn_weights
        elif return_features:
            return logits, fusion_features
        else:
            return logits

    def fit(self, X_train_spatial, y_train,
            epochs=50, batch_size=16, lr=0.001, device='cpu',
            verbose=True, progress_callback=None,
            X_train_temporal=None,
            X_val_spatial=None, X_val_temporal=None, y_val=None,
            use_scheduler=True, scheduler_factor=0.5, scheduler_patience=5, scheduler_min_lr=1e-6):
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
            pass
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
        # 确保标签转换为float类型（BCELoss需要）
        y_train_arr = np.array(y_train, dtype=np.float32)
        if use_dual_input or X_train_temporal is not None:
            # 创建包含空间和时间数据的数据集
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
        
        # 优化器
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.98))

        # 学习率调度器：ReduceLROnPlateau（验证集 Loss 不下降时降低学习率）
        scheduler = None
        if use_scheduler and X_val_spatial is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=scheduler_factor,
                patience=scheduler_patience, min_lr=scheduler_min_lr
            )
        
        # Loss function: 二元交叉熵损失（论文IV.E节）
        # 使用BCELoss因为模型输出层已经包含sigmoid
        criterion = nn.BCELoss()
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
        }

        # 检查是否需要双输入
        dual_input = X_train_temporal is not None

        # 创建 epoch 进度条
        epoch_iterator = tqdm(range(epochs), desc="Training", unit="epoch")

        # 获取 logger 用于记录训练日志
        from utils.logger_config import setup_logger
        train_logger = setup_logger(__name__)

        for epoch in epoch_iterator:
            self.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            all_train_preds = []
            all_train_targets = []

            # 训练批次进度条
            batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, unit="batch")

            for batch_data in batch_iterator:
                if dual_input:
                    # 双输入：(spatial, temporal, labels)
                    batch_x_spatial, batch_x_temporal, batch_y = batch_data
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
                loss = criterion(outputs.squeeze(-1), batch_y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_y.size(0)
                # sigmoid输出概率值，使用阈值0.5判断类别
                preds = (outputs.squeeze(-1) >= 0.5).float()
                epoch_total += batch_y.size(0)
                epoch_correct += (preds == batch_y).sum().item()

                # 收集 predictions 和 targets 用于计算 F1 分数
                all_train_preds.extend(preds.cpu().numpy())
                all_train_targets.extend(batch_y.cpu().numpy())

            epoch_loss /= epoch_total
            epoch_acc = epoch_correct / epoch_total

            # F1 分数（需要 sklearn）
            from sklearn.metrics import f1_score, confusion_matrix
            train_f1 = 100. * f1_score(all_train_targets, all_train_preds, average='binary', zero_division=0)

            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            history['train_f1'].append(train_f1)

            # 验证
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
                        torch.FloatTensor(X_val_spatial),
                        torch.FloatTensor(y_val_arr)
                    )

                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                val_loss = 0
                val_correct = 0
                val_total = 0
                all_val_preds = []
                all_val_targets = []

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
                    # 记录当前学习率
                    current_lr = optimizer.param_groups[0]['lr']
                    train_logger.info(f"  Learning rate: {current_lr:.6f}")

                if verbose:
                    # 计算 TP/TN/FP/FN
                    tn_train, fp_train, fn_train, tp_train = confusion_matrix(all_train_targets, all_train_preds).ravel()
                    tn, fp, fn, tp = confusion_matrix(all_val_targets, all_val_preds).ravel()
                    
                    # 计算 Precision 和 Recall
                    prec_train = tp_train / (tp_train + fp_train) if (tp_train + fp_train) > 0 else 0
                    rec_train = tp_train / (tp_train + fn_train) if (tp_train + fn_train) > 0 else 0
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0

                    # 分多行打印训练集和验证集指标
                    log_lines = [
                        f"Epoch {epoch+1}/{epochs}:",
                        f" Train:  TP={tp_train}, TN={tn_train}, FP={fp_train}, FN={fn_train}",
                        f"         Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, Pre={prec_train:.4f}, Rec={rec_train:.4f}, F1={train_f1:.2f}%",
                        f" Val:    TP={tp}, TN={tn}, FP={fp}, FN={fn}",
                        f"         Loss={val_loss:.4f}, Acc={val_acc:.4f}, Pre={prec:.4f}, Rec={rec:.4f}, F1={val_f1:.2f}%",
                    ]
                    for line in log_lines:
                        tqdm.write(line)
                    # 记录到日志文件（保持相同格式）
                    for line in log_lines:
                        train_logger.info(line)

                    # 更新 epoch 进度条描述
                    epoch_iterator.set_description(f"Training [Val Loss={val_loss:.4f}, F1={val_f1:.2f}%]")
            else:
                if verbose:
                    tn_train, fp_train, fn_train, tp_train = confusion_matrix(all_train_targets, all_train_preds).ravel()
                    prec_train = tp_train / (tp_train + fp_train) if (tp_train + fp_train) > 0 else 0
                    rec_train = tp_train / (tp_train + fn_train) if (tp_train + fn_train) > 0 else 0

                    log_lines = [
                        f"Epoch {epoch+1}/{epochs}:",
                        f"  Train: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, Pre={prec_train:.4f}, Rec={rec_train:.4f}, F1={train_f1:.2f}%",
                        f"         TP={tp_train}, TN={tn_train}, FP={fp_train}, FN={fn_train}",
                    ]
                    for line in log_lines:
                        tqdm.write(line)
                    # 记录到日志文件（保持相同格式）
                    for line in log_lines:
                        train_logger.info(line)

                    epoch_iterator.set_description(f"Training [Loss={epoch_loss:.4f}]")

            # 调用进度回调函数（如提供）
            if progress_callback:
                progress_callback(epoch + 1, epochs)
        
        return history

    def evaluate(self, X_test, y_test, device='cpu', X_test_3d=None):
        """Evaluate model on test set.

        参数:
            X_test: Test features (2D or 3D)
            y_test: Test labels
            device: Device to use
            X_test_3d: Optional 3D test data for temporal models
        """
        self.to(device)
        self.eval()

        # 确定是否有双输入
        if X_test_3d is not None:
            # 使用双输入 (空间 2D + 时间 3D)
            X_test_spatial = X_test
            X_test_temporal = X_test_3d

            y_test_arr = np.array(y_test, dtype=np.float32)
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test_spatial),
                torch.FloatTensor(X_test_temporal),
                torch.FloatTensor(y_test_arr)  # BCELoss 需要 FloatTensor（论文 IV.E）
            )
        else:
            # 自动检测：如果 X_test 是 3D，使用它
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
                # 单输入
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
                    # 双输入
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
                probs = outputs.squeeze(-1)  # (batch, 1) -> (batch,)
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

