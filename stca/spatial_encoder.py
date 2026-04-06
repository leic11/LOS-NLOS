# models/STCA/spatial_encoder.py
"""
Spatial Encoder Module (空间编码器模块) - PyTorch Version
=============================================================

本模块是STCA模型的"第一关"，实现AAM (Attention Aggregation Module) 注意力聚合模块。
根据STCA论文，AAM使用多头自注意力机制对GNSS卫星观测数据进行空间域特征提取。

══════════════════════════════════════════════════════════════════════════════
                              AAM 模块设计理念
══════════════════════════════════════════════════════════════════════════════

为什么需要AAM (注意力聚合模块):
    1. 传统MLP缺乏对卫星间关系的建模能力
       - GNSS NLOS检测需要考虑多卫星之间的几何关系
       - 不同卫星的信号质量相互关联

    2. 注意力机制的优势
       - 动态权重: 根据输入数据自动学习哪些卫星更重要
       - 可解释性: 注意力权重可以可视化显示关注区域
       - 长距离依赖: 轻松捕捉任意卫星之间的关系

    3. AAM的核心思想
       - 首先将输入特征投影到嵌入空间
       - 使用多头自注意力建模卫星间的相关性
       - 通过前馈网络进一步变换
       - 产生空间特征表示

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
    
    使用TransformerEncoder实现多头自注意力，符合STCA论文。
    
    Input:  
        - (batch_size, input_dim) - 2D输入（单卫星）
        - (batch_size, num_satellites, input_dim) - 3D输入（同一历元多卫星）
        
    Output: 
        - (batch_size, num_satellites, embed_dim) - 始终输出3D
    
    论文最优超参数 (Table III):
        - num_heads: 1
        - dropout_rate: 0.5
        - d_ff: embed_dim * 2
        - num_layers: 1
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
    
    def get_config(self):
        """返回模型配置字典"""
        return {
            "input_dim": self.input_dim,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout_rate,
            "encoder_type": "AAM",
        }


# 单元测试
if __name__ == "__main__":
    # 测试 AAM SpatialEncoder - 符合论文设计
    # 论文输入: 4个特征 (C/N0, Elevation, Azimuth, Pseudorange_residual)
    encoder = SpatialEncoder(
        input_dim=4,      # 论文规定的4个特征
        embed_dim=64,     # 论文最优
        num_heads=1,      # 论文最优
        num_layers=1,    # 论文最优
        dropout_rate=0.5  # 论文最优
    )
    
    # 测试2D输入 (单卫星)
    x_2d = torch.randn(32, 4)
    out_2d = encoder(x_2d)
    print(f'2D Input: {x_2d.shape} -> Output: {out_2d.shape}')
    assert out_2d.shape == (32, 1, 64), "2D input should output (batch, 1, embed_dim)"
    
    # 测试3D输入 (同一历元的多个卫星, num_satellites=10)
    x_3d = torch.randn(16, 10, 4)  # batch=16, num_satellites=10, features=4
    out_3d = encoder(x_3d)
    print(f'3D Input: {x_3d.shape} -> Output: {out_3d.shape}')
    assert out_3d.shape == (16, 10, 64), "3D input should output (batch, num_satellites, embed_dim)"
    
    # 打印模型结构
    print("\n模型结构:")
    print(encoder)
    
    # 打印配置
    print("\n模型配置:")
    print(encoder.get_config())
    
    print('\n✓ AAM SpatialEncoder 单元测试通过!')
    print('  - 输入特征维度: 4 (C/N0, Elevation, Azimuth, Pseudorange_residual)')
    print('  - num_satellites 表示同一历元捕获的卫星数量')
    print('  - 默认超参数: num_heads=1, dropout_rate=0.5, num_layers=1')