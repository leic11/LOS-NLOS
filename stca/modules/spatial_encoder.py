# models/STCA/spatial_encoder.py
"""
Spatial Encoder Module (空间编码器模块) - PyTorch Version
=============================================================

本模块是STCA模型的"第一关"，实现AAM (Attention Aggregation Module) 注意力聚合模块。
根据STCA论文，AAM使用多头自注意力机制对GNSS卫星观测数据进行空间域特征提取。

══════════════════════════════════════════════════════════════════════════════
                              数学公式定义
══════════════════════════════════════════════════════════════════════════════

根据STCA论文 (公式9-16)，AAM的数学公式:

输入投影:
    H = XW + b    (线性变换到嵌入空间)
    
多头自注意力 (Multi-Head Self-Attention):
    head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

完整前向传播:
    A = MultiHeadSelfAttention(H)
    H' = LayerNorm(H + A)     (残差连接 + LayerNorm)
    F = FFN(H')
    H_spatial = LayerNorm(H' + F)  (残差连接 + LayerNorm)

══════════════════════════════════════════════════════════════════════════════
                              输入输出规格
══════════════════════════════════════════════════════════════════════════════

输入格式:
    - 2D输入: (batch_size, input_dim) - 单历元单卫星
    - 3D输入: (batch_size, num_satellites, input_dim) - 当前历元的所有卫星
      * 注意: seq_len 表示同一历元捕获的卫星数量（如 max_satellites=30）
      * 不是时间窗口长度！

输出格式:
    - 始终为3D: (batch_size, num_satellites, embed_dim)
    - num_satellites: 对于2D输入自动扩展为1

══════════════════════════════════════════════════════════════════════════════
                              AAM 网络架构
══════════════════════════════════════════════════════════════════════════════

    输入 X (batch, num_satellites, input_dim)
           │
           ▼
    ┌─────────────────────────────────────┐
    │  输入投影层 (Input Projection)       │
    │  Linear(input_dim, embed_dim)       │
    └─────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────┐
    │  MultiheadSelfAttention (MHA)        │
    │  Q=K=V=投影后特征                   │
    │  num_heads注意力头                   │
    └─────────────────────────────────────┘
           │ 残差连接 + LayerNorm
           ▼
    ┌─────────────────────────────────────┐
    │  前馈网络 (FFN)                     │
    │  Linear → ReLU → Linear             │
    └─────────────────────────────────────┘
           │ 残差连接 + LayerNorm
           ▼
    输出 (batch, num_satellites, embed_dim)

Related Modules:
    - TemporalEncoder: 与AAM并行工作，接收相同原始输入进行时序建模
    - CrossAttention: 接收AAM和TemporalEncoder的输出进行跨域特征融合
    - STCAModel: 集成AAM和TemporalEncoder作为双路并行结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialEncoder(nn.Module):
    """
    AAM (Attention Aggregation Module) based Spatial Encoder.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        num_heads: int = 1,        # 论文最优: 1
        num_layers: int = 1,
        d_ff: int = None,
        dropout_rate: float = 0.5, # 论文最优: 0.5
        activation: str = "relu",
    ):
        """
        Args:
            input_dim: 输入特征维度 (4: C/N0, Elevation, Azimuth, Pseudorange_residual)
            embed_dim: 输出嵌入维度 (论文最优: 64)
            num_heads: 注意力头数 (论文最优: 1)
            num_layers: 编码器层数 (论文最优: 1)
            d_ff: 前馈网络隐藏层维度 (默认 embed_dim * 2 = 128)
            dropout_rate: Dropout比率 (论文最优: 0.5)
            activation: FFN激活函数 ("relu", "gelu")
        """
        super(SpatialEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff if d_ff is not None else embed_dim * 2
        self.dropout_rate = dropout_rate
        
        # 验证embed_dim能被num_heads整除
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        # 1. 输入投影层 (Input Projection)
        # 将原始特征映射到嵌入空间
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
            # 2. 使用 TransformerEncoder 实现多层堆叠
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=self.d_ff,
            dropout=dropout_rate,
            activation=activation if activation != "gelu" else "gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 初始化输入投影层
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        # TransformerEncoder 已有内置初始化，无需额外处理
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
                - 2D: (batch_size, input_dim) - 单卫星
                - 3D: (batch_size, num_satellites, input_dim) - 多卫星（同一历元）
                * 注意: num_satellites 是同一历元捕获的卫星数量，不是时间窗口长度
            
        Returns:
            空间嵌入向量, 形状为 (batch_size, num_satellites, embed_dim)
        """
        # 处理2D输入 (batch, features) -> 扩展为3D (batch, 1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Step 1: 输入投影
        # (batch, seq_len, input_dim) -> (batch, seq_len, embed_dim)
        h = self.input_proj(x)
        
        # Step 2: 多层 Transformer Encoder
        # 包含自注意力 + 残差连接 + LayerNorm + FFN + 残差连接 + LayerNorm
        h = self.transformer_encoder(h)
        
        # 输出: (batch, seq_len, embed_dim)
        return h

