# models/STCA/sparse_representation.py
"""
Sparse Representation Module (稀疏表示模块) - PyTorch Version
=============================================================

本模块实现论文第三节 F 的「可学习稀疏正则化器」作为激活函数 Ω(z)。

论文公式参考 (III-F 节):
    Ω(z) = 
        { w₂(z-b₂) + w₁(b₂-b₁),                     b₂ ≤ z
        { w₁(z-b₁),                                 b₁ ≤ z < b₂
        { 0,                                        -b₁ ≤ z < b₁
        { w₁(z+b₁),                                 -b₂ ≤ z < -b₁
        { w₂(z+b₂) + w₁(b₁-b₂),                     z < -b₂

特点：
    - Ω(z) 是奇函数：Ω(-z) = -Ω(z)
    - 在 [-b₁, b₁] 区间输出 0，实现稀疏性
    - 参数约束: w₁ > 0, w₂ > 0, 0 ≤ b₁ ≤ b₂
"""

from .constants import SPARSE_EMBED_DIM

import torch
import torch.nn as nn
import torch.nn.functional as F


def _omega_piecewise(z: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, 
                     b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """
    可学习稀疏正则化激活函数 Ω(z)，逐元素计算，保留符号信息。
    Ω(z) 是奇函数：Ω(-z) = -Ω(z)
    
    参数约束:
        - w1 > 0, w2 > 0
        - 0 ≤ b1 ≤ b2
    
    分段区间:
        - z ≥ b2:        Ω(z) = w2*(z-b2) + w1*(b2-b1)
        - b1 ≤ z < b2:    Ω(z) = w1*(z-b1)
        - -b1 ≤ z < b1:    Ω(z) = 0  (稀疏性来源)
        - -b2 ≤ z < -b1:  Ω(z) = w1*(z+b1) = -w1*(b1-|z|)
        - z < -b2:        Ω(z) = w2*(z+b2) + w1*(b1-b2) = -[w2*(|z|-b2) + w1*(b2-b1)]
    """
    # 确保参数形状正确
    w1 = w1.view(-1)
    w2 = w2.view(-1)
    b1 = b1.view(-1)
    b2 = b2.view(-1)
    
    # 分段判断（基于 z 本身，保留符号）
    in_inner_pos = (z >= b1) & (z < b2)           # b1 ≤ z < b2
    in_outer_pos = z >= b2                        # z ≥ b2
    in_inner_neg = (z >= -b2) & (z < -b1)         # -b2 ≤ z < -b1
    in_outer_neg = z < -b2                        # z < -b2
    # 其余情况: -b1 ≤ z < b1 -> Ω(z) = 0
    
    out = torch.zeros_like(z)
    
    # 正值区域
    # b1 ≤ z < b2: Ω(z) = w1 * (z - b1)
    inner_pos_val = w1 * (z - b1)
    out = torch.where(in_inner_pos, inner_pos_val, out)
    
    # z ≥ b2: Ω(z) = w2 * (z - b2) + w1 * (b2 - b1)
    outer_pos_val = w2 * (z - b2) + w1 * (b2 - b1)
    out = torch.where(in_outer_pos, outer_pos_val, out)
    
    # 负值区域（奇函数性质，自动处理符号）
    # -b2 ≤ z < -b1: Ω(z) = w1 * (z + b1)
    inner_neg_val = w1 * (z + b1)
    out = torch.where(in_inner_neg, inner_neg_val, out)
    
    # z < -b2: Ω(z) = w2 * (z + b2) + w1 * (b1 - b2)
    outer_neg_val = w2 * (z + b2) + w1 * (b1 - b2)
    out = torch.where(in_outer_neg, outer_neg_val, out)
    
    return out


class SparseRegularizer(nn.Module):
    """
    可学习稀疏正则化器 Ω(z)，作为激活函数使用。
    
    对融合特征 Z 施加 3 次 Ω(z) 运算，上一层输出作为下一层输入，
    使特征向量中大部分元素趋近于 0，实现稀疏表示。
    
    Input:  (batch_size, embed_dim) - 交叉注意力输出的融合特征 Z
    Output: (batch_size, embed_dim) - 稀疏化后的特征
    """
    
    def __init__(
        self,
        embed_dim: int = None,
    ):
        """
        Args:
            embed_dim: 特征维度，应与交叉注意力输出维度一致（论文中为64）
        """
        super(SparseRegularizer, self).__init__()
        self.embed_dim = embed_dim
        
        # 可学习参数 w1, w2, b1, b2，每维一组
        # 约束: w1 > 0, w2 > 0, 0 <= b1 <= b2
        # 使用 softplus 保证 w > 0，使用 sigmoid 保证 0 <= b1 <= b2
        
        # w1_raw, w2_raw: 通过 softplus 映射到 (0, +inf)
        self.w1_raw = nn.Parameter(torch.ones(embed_dim) * 1.0)
        self.w2_raw = nn.Parameter(torch.ones(embed_dim) * 0.5)

        # b1_raw, b2_raw: 通过约束映射
        # b1 = b2 * sigmoid(b1_raw), 保证 0 <= b1 <= b2
        # 减小初始值以允许更小的输入通过
        self.b2_raw = nn.Parameter(torch.ones(embed_dim) * -2.0)  # softplus(-2) ≈ 0.127
        self.b1_raw = nn.Parameter(torch.zeros(embed_dim))  # sigmoid(0) = 0.5, b1 = 0.5 * b2
    
    def _get_parameters(self):
        """返回满足约束的参数 (w1, w2, b1, b2)"""
        w1 = F.softplus(self.w1_raw)   # w1 > 0
        w2 = F.softplus(self.w2_raw)   # w2 > 0
        b2 = F.softplus(self.b2_raw)   # b2 > 0
        b1 = b2 * torch.sigmoid(self.b1_raw)  # 0 <= b1 <= b2
        return w1, w2, b1, b2
    
    def _omega_layer(self, z: torch.Tensor) -> torch.Tensor:
        """对特征 z 施加 Ω(z) 激活函数"""
        w1, w2, b1, b2 = self._get_parameters()
        return _omega_piecewise(z, w1, w2, b1, b2)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        对输入特征 z 施加 3 次稀疏正则化
        
        Args:
            z: 融合特征，形状 (batch_size, embed_dim)
            
        Returns:
            稀疏化后的特征，形状 (batch_size, embed_dim)
        """
        # 第一次 Ω(z)
        z = self._omega_layer(z)
        # 第二次 Ω(z)
        z = self._omega_layer(z)
        # 第三次 Ω(z)
        z = self._omega_layer(z)
        return z


class SparseRepresentation(nn.Module):
    """
    稀疏表示模块 - 直接对输入特征进行稀疏化
    
    注意：论文中稀疏正则化直接作用于交叉注意力的输出 Z，
    不需要额外的线性投影层。此模块直接接受 Z 并输出稀疏化后的特征。
    
    Input: (batch_size, embed_dim) - 来自交叉注意力的融合特征 Z
    Output: (batch_size, embed_dim) - 稀疏表示后的特征
    """
    
    def __init__(
        self,
        embed_dim: int = None,
    ):
        """
        Args:
            embed_dim: 特征维度，应与交叉注意力输出维度一致（论文中为64）
        """
        super(SparseRepresentation, self).__init__()
        # 使用常量默认参数
        self.embed_dim = embed_dim if embed_dim is not None else SPARSE_EMBED_DIM
        
        # 直接使用 SparseRegularizer，不添加额外投影层
        self.sparse_regularizer = SparseRegularizer(embed_dim=self.embed_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        对输入特征 z 进行稀疏化处理
        
        Args:
            z: 融合特征 Z，来自交叉注意力层
               - 形状 (batch_size, embed_dim) 或 (batch_size, 1, embed_dim)
               
        Returns:
            稀疏化后的特征，形状 (batch_size, embed_dim)
        """
        # 处理维度：如果是 (batch, 1, embed_dim)，压缩到 (batch, embed_dim)
        if z.dim() == 3 and z.size(1) == 1:
            z = z.squeeze(1)  # (batch, embed_dim)
        
        # 稀疏正则化处理
        z = self.sparse_regularizer(z)

        return z
