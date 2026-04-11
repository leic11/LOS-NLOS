from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSparseReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.b1 = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(0.03, dtype=torch.float32))

    def forward(self, z):
        absz = z.abs()
        w1 = F.softplus(self.w1).clamp(min=1e-3)
        w2 = F.softplus(self.w2).clamp(min=1e-3)
        b1 = F.softplus(self.b1).clamp(min=1e-4)
        delta = F.softplus(self.delta).clamp(min=1e-4)
        b2 = b1 + delta

        reg = torch.zeros_like(absz)
        c1 = absz >= b2
        c2 = (absz < b2) & (absz >= b1)
        reg[c1] = w2 * (absz[c1] - b2) + w1 * (b2 - b1)
        reg[c2] = w1 * (absz[c2] - b1)
        return reg.mean()

    def prox(self, z):
        absz = z.abs()
        b1 = F.softplus(self.b1).clamp(min=1e-4)
        delta = F.softplus(self.delta).clamp(min=1e-4)
        b2 = b1 + delta

        prox_z = z.clone()
        mask_small = absz < b1
        mask_middle = (absz >= b1) & (absz < b2)
        mask_large = absz >= b2

        prox_z[mask_small] = 0
        prox_z[mask_middle] = (absz[mask_middle] - b1) * (z[mask_middle] / (absz[mask_middle] + 1e-12))
        prox_z[mask_large] = (absz[mask_large] - b2) * (z[mask_large] / (absz[mask_large] + 1e-12))
        return prox_z


class LOSNLOSModel(nn.Module):
    def __init__(
        self,
        lstm_hidden=64,
        lstm_layers=2,
        attn_hidden=64,
        attn_heads=1,
        ff_hidden=128,
        dropout=0.6,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(9, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.spatial_embed = nn.Linear(9, attn_hidden)
        self.self_attn = nn.MultiheadAttention(embed_dim=attn_hidden, num_heads=attn_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(attn_hidden, ff_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, attn_hidden),
        )
        self.ln1 = nn.LayerNorm(attn_hidden)
        self.ln2 = nn.LayerNorm(attn_hidden)

        self.cross_attn = nn.MultiheadAttention(embed_dim=attn_hidden, num_heads=attn_heads, dropout=dropout)
        self.cross_ln = nn.LayerNorm(attn_hidden)

        self.regularizer = LearnableSparseReg()
        self.classifier = nn.Linear(attn_hidden, 2)

    def forward(self, time_input, space_input, space_mask=None, return_fused=False):
        x = time_input.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        time_feat = h_n[-1]

        se = self.spatial_embed(space_input)
        se_t = se.transpose(0, 1)
        sa, _ = self.self_attn(se_t, se_t, se_t, key_padding_mask=space_mask)
        sa = sa.transpose(0, 1)
        sp1 = self.ln1(se + sa)
        sp2 = self.ffn(sp1)
        sp = self.ln2(sp1 + sp2)

        q = time_feat.unsqueeze(0)
        sp_t = sp.transpose(0, 1)
        co, _ = self.cross_attn(q, sp_t, sp_t, key_padding_mask=space_mask)
        co = co.squeeze(0)
        fused = self.cross_ln(co + time_feat)

        reg_loss = self.regularizer(fused)
        logits = self.classifier(fused)

        if return_fused:
            return logits, reg_loss, fused
        return logits, reg_loss
