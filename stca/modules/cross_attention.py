# models/STCA/cross_attention.py
"""
Cross-Attention Module (交叉注意力模块) - PyTorch Version
=============================================================

本模块实现了多头交叉注意力(Multi-Head Cross-Attention)机制，是STCA模型中
实现空间-时间特征融合的核心组件。在GNSS NLOS检测任务中，交叉注意力机制
发挥以下关键作用:

1. 【跨域特征融合】
   - 将空间编码器提取的空间特征(卫星几何分布、信号质量等)与时间编码器
     提取的时序特征(信号变化趋势、历史模式等)进行深度融合
   - 使模型能够同时学习到"某一时刻的卫星分布状态"和"信号的时间演变规律"

2. 【自适应特征加权】
   - 通过注意力权重自动学习不同时间步或空间位置的重要性
   - 重要的时间步或空间位置会获得更高的注意力权重

3. 【增强模型可解释性】
   - 返回的注意力权重可以可视化，帮助理解模型的决策依据
   - 分析注意力权重可以揭示哪些卫星对当前预测贡献最大

══════════════════════════════════════════════════════════════════════════════
                              在GNSS NLOS检测中的具体应用
══════════════════════════════════════════════════════════════════════════════

典型应用场景:
    - Query: 时间特征 h_t (来自LSTM-TFE)，形状 (batch, 1, embed_dim)
    - Key/Value: 空间特征 h_s (来自AAM)，形状 (batch, num_satellites, embed_dim)
    
    论文公式 (17):
        Q~ = W~q · h_t + b~q    (时间特征作为Query)
        K~ = W~k · h_s + b~k    (空间特征作为Key)
        V~ = W~v · h_s + b~v    (空间特征作为Value)
    
    这种设计使"时间特征查询空间特征"，实现时空信息的融合

输入输出说明:
    Input Shape:
        - query: (batch_size, 1, query_dim) - 时间特征 h_t
        - key:   (batch_size, num_satellites, kv_dim) - 空间特征 h_s
        - value: (batch_size, num_satellites, kv_dim) - 空间特征 h_s

    Output Shape:
        - (batch_size, 1, embed_dim) - 注意力增强后的特征表示

══════════════════════════════════════════════════════════════════════════════
                              数学公式定义 (多头交叉注意力)
══════════════════════════════════════════════════════════════════════════════

缩放点积注意力 (Scaled Dot-Product Attention):
    Attention(Q, K, V) = softmax( (Q × K^T) / √d_k ) × V
    
    其中:
    - Q ∈ R^(N_q, d): 查询矩阵 (Query)
    - K ∈ R^(N_k, d): 键矩阵 (Key)
    - V ∈ R^(N_v, d): 值矩阵 (Value)
    - d: 键/值的维度
    - √d_k: 缩放因子，用于防止梯度消失
    - softmax: 归一化指数函数，确保权重和为1

线性投影生成Q, K, V:
    Q = X × W_Q
    K = X × W_K  
    V = X × W_V
    
    其中:
    - W_Q, W_K, W_V ∈ R^(d_model, d): 可学习的投影矩阵

多头注意力 (Multi-Head Attention):
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
    
    其中:
    - head_i = Attention(Q × W_i^Q, K × W_i^K, V × W_i^V)
    - h: 注意力头数
    - W_O ∈ R^(h×d_v, d_model): 输出投影矩阵

交叉注意力 (Cross-Attention):
    交叉注意力与自注意力的区别在于Q来自一个序列，而K和V来自另一个序列:
    - Q = Temporal_Feature × W_Q    (时间特征作为Query)
    - K = Spatial_Feature × W_K     (空间特征作为Key)
    - V = Spatial_Feature × W_V     (空间特征作为Value)

    这种设计使"时间特征查询空间特征"，实现时空信息的融合

论文最优超参数 (Table III):
    - num_heads: 1
    - dropout_rate: 0.5
    - embed_dim: 64

Related Modules:
    - TemporalEncoder: 提供Query特征 (h_t)
    - SpatialEncoder: 提供Key/Value特征 (h_s)
    - STCAModel: 集成此模块进行时空特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention layer (多头交叉注意力层).
    
    实现跨域特征融合，时间特征作为Query查询空间特征。
    符合STCA论文公式 (17) 的设计。
    
    Input:
        query: (batch_size, 1, query_dim) - 时间特征 h_t (来自LSTM-TFE)
        key:   (batch_size, num_satellites, kv_dim) - 空间特征 h_s (来自AAM)
        value: (batch_size, num_satellites, kv_dim) - 空间特征 h_s
    Output:
        (batch_size, 1, embed_dim) - 注意力增强后的特征表示
    
    论文公式 (17):
        Q~ = W~q · h_t + b~q    (时间特征作为Query)
        K~ = W~k · h_s + b~k    (空间特征作为Key)
        V~ = W~v · h_s + b~v    (空间特征作为Value)
    
    论文最优超参数 (Table III):
        - num_heads: 1
        - dropout_rate: 0.5
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 1,      # 论文最优: 1
        dropout_rate: float = 0.5,  # 论文最优: 0.5
    ):
        """
        Args:
            embed_dim: 注意力输出的嵌入维度 (论文最优: 64)
            num_heads: 注意力头数 (论文最优: 1)
            dropout_rate: Dropout比率 (论文最优: 0.5)
        """
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.query_dim = embed_dim
        self.kv_dim = embed_dim
        
        self.depth = embed_dim // num_heads
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        # Q, K, V 投影层
        self.query_proj = nn.Linear(self.query_dim, embed_dim)
        self.key_proj = nn.Linear(self.kv_dim, embed_dim)
        self.value_proj = nn.Linear(self.kv_dim, embed_dim)
        
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 缩放因子
        self.scale = math.sqrt(self.depth)
    
    def forward(self, query, key, value, return_attention_weights=False):
        """
        前向传播
        
        Args:
            query: 查询张量 (batch, 1, embed_dim) - 时间特征 h_t
            key: 键张量 (batch, num_satellites, embed_dim) - 空间特征 h_s
            value: 值张量 (batch, num_satellites, embed_dim) - 空间特征 h_s
            return_attention_weights: 是否返回注意力权重
            
        Returns:
            output: 注意力增强后的特征 (batch, 1, embed_dim)
            attention_weights: 注意力权重 (可选, shape: batch, 1, num_satellites)
        """
        batch_size = query.size(0)
        
        # 投影 Q, K, V
        Q = self.query_proj(query)  # (batch, query_len, embed_dim)
        K = self.key_proj(key)      # (batch, key_len, embed_dim)
        V = self.value_proj(value)  # (batch, value_len, embed_dim)
        
        # 分割多头
        # 将embed_dim分割成num_heads个深度为depth的维度
        Q = self._split_heads(Q, batch_size)  # (batch, num_heads, query_len, depth)
        K = self._split_heads(K, batch_size)  # (batch, num_heads, key_len, depth)
        V = self._split_heads(V, batch_size)  # (batch, num_heads, value_len, depth)
        
        # 计算注意力分数
        # Q @ K^T / sqrt(d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        # attention_weights @ V
        attended = torch.matmul(attention_weights, V)  # (batch, num_heads, query_len, depth)
        
        # 合并多头
        attended = self._merge_heads(attended, batch_size)  # (batch, query_len, embed_dim)
        
        # 输出投影
        output = self.out_proj(attended)
        output = self.dropout(output)
        
        if return_attention_weights:
            # 返回平均注意力权重 (batch, query_len, key_len)
            avg_attention = attention_weights.mean(dim=1)
            return output, avg_attention
        
        return output
    
    def _split_heads(self, x, batch_size):
        """
        将张量分割成多个注意力头
        
        Args:
            x: (batch, seq_len, embed_dim)
            
        Returns:
            (batch, num_heads, seq_len, depth)
        """
        seq_len = x.size(1)
        x = x.view(batch_size, seq_len, self.num_heads, self.depth)
        x = x.transpose(1, 2)  # (batch, num_heads, seq_len, depth)
        return x
    
    def _merge_heads(self, x, batch_size):
        """
        合并多个注意力头
        
        Args:
            x: (batch, num_heads, seq_len, depth)
            
        Returns:
            (batch, seq_len, embed_dim)
        """
        x = x.transpose(1, 2)  # (batch, seq_len, num_heads, depth)
        x = x.contiguous()
        x = x.view(batch_size, -1, self.embed_dim)
        return x
