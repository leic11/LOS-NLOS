# models/STCA/temporal_encoder.py
"""
Temporal Encoder Module (时序编码器模块) - PyTorch Version
=============================================================

本模块使用LSTM(长短期记忆网络)实现LSTM-TFE (LSTM-based Temporal Feature Extraction Module，
时序特征提取模块)，对目标卫星的历史时间窗口进行时序建模，提取时间维度的特征表示。

══════════════════════════════════════════════════════════════════════════════
                              在GNSS NLOS检测中的具体应用
══════════════════════════════════════════════════════════════════════════════

典型输入: 目标卫星的历史观测特征时间窗口
    - 形状: (batch_size, window_size, input_dim)
    - input_dim: 4 (C/N0, Elevation, Azimuth, Pseudorange_residual)
    - window_size: 时间窗口长度 (如 10)

输出: 最后一个时间步的隐藏状态
    - 形状: (batch_size, embed_dim) 或 (batch_size, embed_dim*2) 如果双向
    - 包含了整个时间窗口的上下文信息

══════════════════════════════════════════════════════════════════════════════
                              数学公式定义 (LSTM)
══════════════════════════════════════════════════════════════════════════════

标准LSTM方程 (引用于论文):

遗忘门 (Forget Gate):
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
    
    决定有多少过去的信息需要遗忘:
    - f_t ∈ [0, 1]: 遗忘门输出
    - σ: sigmoid函数
    - W_f: 遗忘门权重矩阵

输入门 (Input Gate):
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
    
候选记忆 (Candidate Cell State):
    C̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
    
    生成新的候选记忆内容

单元状态更新 (Cell State Update):
    C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
    
    - ⊙: 逐元素乘法 (Hadamard积)
    - 结合遗忘门和输入门的输出更新单元状态

输出门 (Output Gate):
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
    
隐藏状态 (Hidden State):
    h_t = o_t ⊙ tanh(C_t)
    
    - 决定输出到下一个时间步的信息

符号定义:
    - x_t ∈ R^(d_in): t时刻的输入向量
    - h_{t-1} ∈ R^(d_h): t-1时刻的隐藏状态
    - C_{t-1} ∈ R^(d_h): t-1时刻的单元状态
    - f_t, i_t, o_t ∈ R^(d_h): 门控向量
    - C_t, h_t ∈ R^(d_h): 更新后的单元状态和隐藏状态
    - W_* ∈ R^(d_h × (d_h + d_in)): 权重矩阵
    - b_* ∈ R^(d_h): 偏置向量

注意: 论文使用单向LSTM，仅利用历史信息，不利用未来信息

输入输出说明:
    Input Shape:
        - (batch_size, window_size, input_dim) - 目标卫星的历史观测特征
        
    Output Shape:
        - (batch_size, embed_dim) - 编码后的时序表示 (单向)
        - (batch_size, embed_dim*2) - 编码后的时序表示 (双向)

论文最优超参数 (Table III):
    - input_dim: 4 (原始观测特征)
    - hidden_dim: 64
    - num_layers: 1
    - dropout_rate: 0.5
    - bidirectional: False (单向LSTM)

Related Modules:
    - SpatialEncoder: 与本模块并行工作，接收相同原始输入进行空间建模
    - CrossAttention: 接收AAM和TemporalEncoder的输出进行跨域特征融合
    - STCAModel: 集成AAM和TemporalEncoder作为双路并行结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """
    LSTM-based Temporal Encoder (LSTM-TFE Module).
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        num_layers: int = 1,
        dropout_rate: float = 0.5,  # 论文最优: 0.5
        bidirectional: bool = False,  # 论文最优: False (单向LSTM)
    ):
        """
        Args:
            input_dim: 每个时间步的输入特征维度 (论文最优: 4)
            embed_dim: LSTM隐藏维度 (论文最优: 64)
            num_layers: LSTM层数 (论文最优: 1)
            dropout_rate: Dropout比率 (论文最优: 0.5)
            bidirectional: 是否使用双向LSTM (论文最优: False)
        """
        super(TemporalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # 输出维度
        if bidirectional:
            self.output_dim = embed_dim * 2
        else:
            self.output_dim = embed_dim

        # 输出 dropout 层（确保单层 LSTM 也能应用 dropout）
        self.dropout = nn.Dropout(dropout_rate)

        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化LSTM权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # 设置遗忘门偏置为1 (有助于梯度流动)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
    
    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量, 形状为 (batch_size, window_size, input_dim)
                - input_dim: 4 (C/N0, Elevation, Azimuth, Pseudorange_residual)
                - window_size: 时间窗口长度
            return_sequence: 是否返回整个序列 (默认False, 只返回最后隐藏状态)
            
        Returns:
            时序表示向量:
            - 如果 return_sequence=False: (batch_size, output_dim)
            - 如果 return_sequence=True: (batch_size, window_size, output_dim)
        """
        # LSTM前向传播
        # output: (batch, seq_len, num_directions * hidden_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        output, (hidden, cell) = self.lstm(x)
        
        if return_sequence:
            # 返回整个序列
            return output
        else:
            # 返回最后一个时间步的隐藏状态
            # 如果是双向，取前向和后向最后一个隐藏状态的拼接
            if self.bidirectional:
                # hidden: (num_layers * 2, batch, hidden_size)
                # 取最后一层的前向和后向隐藏状态
                forward_h = hidden[-2, :, :]  # (batch, hidden_size)
                backward_h = hidden[-1, :, :]  # (batch, hidden_size)
                # 拼接
                last_hidden = torch.cat([forward_h, backward_h], dim=1)
            else:
                # 取最后一层的隐藏状态
                last_hidden = hidden[-1, :, :]  # (batch, hidden_size)

            # 应用输出 dropout
            return self.dropout(last_hidden)
