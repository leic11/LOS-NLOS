# modules/cross_attention.py
"""
Multi-Head Cross-Attention Module (多头交叉注意力模块) - PyTorch Version
=========================================================================

本模块实现 STCA 论文中的交叉注意力融合机制。

论文公式 (17):
    Attention(Q, K, V) = softmax(QK^T / √d_k)V

其中:
    - Q = W_q · h_t + b_q (时间特征作为 Query)
    - K = W_k · h_s + b_k (空间特征作为 Key)
    - V = W_v · h_s + b_v (空间特征作为 Value)

输入输出规格:
    Input:
        query: (batch_size, 1, query_input_dim) - 时间特征 h_t
        key: (batch_size, num_satellites, kv_input_dim) - 空间特征 h_s
        value: (batch_size, num_satellites, kv_input_dim) - 空间特征 h_s

    Output:
        (batch_size, 1, embed_dim) - 注意力增强后的特征表示

Related Modules:
    - SpatialEncoder: 提供空间特征 h_s
    - TemporalEncoder: 提供时间特征 h_t
    - STCAModel: 集成交叉注意力作为特征融合模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention layer (多头交叉注意力层).

    实现跨域特征融合，时间特征作为 Query 查询空间特征。
    符合 STCA 论文公式 (17) 的设计。

    Input:
        query: (batch_size, 1, query_input_dim) - 时间特征 h_t (来自 LSTM-TFE)
        key:   (batch_size, num_satellites, kv_input_dim) - 空间特征 h_s (来自 AAM)
        value: (batch_size, num_satellites, kv_input_dim) - 空间特征 h_s
    Output:
        (batch_size, 1, embed_dim) - 注意力增强后的特征表示

    论文公式 (17):
        Q~ = W~q · h_t + b~q    (时间特征作为 Query)
        K~ = W~k · h_s + b~k    (空间特征作为 Key)
        V~ = W~v · h_s + b~v    (空间特征作为 Value)

    论文最优超参数 (Table III):
        - num_heads: 1
        - dropout_rate: 0.5
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 1,      # 论文最优：1
        dropout_rate: float = 0.5,  # 论文最优：0.5
        query_input_dim: int = None,   # Query 输入维度（默认等于 embed_dim）
        kv_input_dim: int = None,      # Key/Value 输入维度（默认等于 embed_dim）
    ):
        """
        Args:
            embed_dim: 注意力输出的嵌入维度 (论文最优：64)
            num_heads: 注意力头数 (论文最优：1)
            dropout_rate: Dropout 比率 (论文最优：0.5)
            query_input_dim: Query 输入维度（默认等于 embed_dim）
            kv_input_dim: Key/Value 输入维度（默认等于 embed_dim）
        """
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.query_input_dim = query_input_dim if query_input_dim is not None else embed_dim
        self.kv_input_dim = kv_input_dim if kv_input_dim is not None else embed_dim

        self.depth = embed_dim // num_heads

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        # Q, K, V 投影层（使用输入维度）
        self.query_proj = nn.Linear(self.query_input_dim, embed_dim)
        self.key_proj = nn.Linear(self.kv_input_dim, embed_dim)
        self.value_proj = nn.Linear(self.kv_input_dim, embed_dim)

        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout 层
        self.dropout = nn.Dropout(dropout_rate)

        # 缩放因子
        self.scale = math.sqrt(self.depth)

    def forward(self, query, key, value, return_attention_weights=False):
        """
        前向传播

        Args:
            query: 查询张量 (batch, 1, query_input_dim) - 时间特征 h_t
            key: 键张量 (batch, num_satellites, kv_input_dim) - 空间特征 h_s
            value: 值张量 (batch, num_satellites, kv_input_dim) - 空间特征 h_s
            return_attention_weights: 是否返回注意力权重

        Returns:
            output: 注意力增强后的特征 (batch, 1, embed_dim)
            attention_weights: 注意力权重 (可选，shape: batch, 1, num_satellites)
        """
        batch_size = query.size(0)

        # 投影 Q, K, V
        Q = self.query_proj(query)  # (batch, query_len, embed_dim)
        K = self.key_proj(key)      # (batch, key_len, embed_dim)
        V = self.value_proj(value)  # (batch, value_len, embed_dim)

        # 分割多头
        # 将 embed_dim 分割成 num_heads 个深度为 depth 的维度
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
        seq_len = x.size(2)
        x = x.transpose(1, 2).contiguous()  # (batch, seq_len, num_heads, depth)
        x = x.view(batch_size, seq_len, self.embed_dim)
        return x
