# -*- coding: utf-8 -*-
"""
创新点一核心模型: ESM-2 (1D 序列语言模型) x E(n)-等变图神经网络 (3D 空间构象) 交叉注意力融合

模型结构:
    1. ESM-2 冻结表征 (预计算缓存) 作为 1D 语义特征;
    2. 残基级 3D 图 (节点特征 = one-hot + 理化性质 + 投影后的 ESM 语义)
       输入多层 E(n)-等变图神经网络, 坐标随消息传递同步更新 (E(3) 等变),
       提取 3D 空间拓扑特征 —— 捕获 1D 序列上相距甚远但空间相邻的裂解基序;
    3. 双向交叉注意力 (co-attention): ESM 序列特征 <-> EGNN 空间特征互为 Query/Key/Value,
       门控残差融合, 实现"序列-结构"多模态对齐;
    4. 掩码均值 + 最大值池化 -> MLP 分类头 (Softmax 输出, 与 baseline 协议一致)。

全部使用稠密张量 + 掩码实现, 不依赖 torch_geometric。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NODE_FEAT_DIM = 26   # 21 one-hot + 5 理化性质 (与 dataset.py 保持一致)

# ---------------------------------------------------------------------------
# 消融实验预设表 (变体名 -> 模型开关); train.py/test.py/run_ablation.py 共用
# ---------------------------------------------------------------------------
ABLATION_PRESETS = {
    'full':     dict(use_egnn=True,  use_esm=True,
                     use_cross_attn=True,  use_coord_update=True),
    'no_3d':    dict(use_egnn=False, use_esm=True,      # 去掉 3D-EGNN 空间分支
                     use_cross_attn=True,  use_coord_update=True),
    'no_esm':   dict(use_egnn=True,  use_esm=False,     # 去掉 ESM 序列语义
                     use_cross_attn=True,  use_coord_update=True),
    'no_attn':  dict(use_egnn=True,  use_esm=True,      # 交叉注意力 -> 等权平均
                     use_cross_attn=False, use_coord_update=True),
    'no_coord': dict(use_egnn=True,  use_esm=True,      # 去掉等变坐标更新
                     use_cross_attn=True,  use_coord_update=False),
}


# ---------------------------------------------------------------------------
# E(n)-等变图神经网络层 (稠密掩码实现)
# ---------------------------------------------------------------------------

class EGNNLayer(nn.Module):
    """
    E(n)-Equivariant GNN 层 (Satorras et al., ICML 2021) 的稠密矩阵形式。
    节点坐标 x 随消息传递同步更新, 对平移/旋转/反射保持等变性。

    Args:
        hidden_dim: 节点特征维度
        edge_feat_dim: 边条件特征维度 (seq_adj 指示 + 距离)
    """

    def __init__(self, hidden_dim=128, dropout=0.1, update_coord=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_coord = update_coord

        # 边消息: [h_i | h_j | dist | seq_flag] -> m_ij
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 坐标更新权重 (标量门控); 消融 no_coord 时整组不创建
        if update_coord:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        # 节点更新 (pre-norm 残差)
        self.norm = nn.LayerNorm(hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, x, adj, seq_adj, mask):
        """
        Args:
            h:       (B, N, H) 节点特征
            x:       (B, N, 3) 节点坐标
            adj:     (B, N, N) 邻接 (0/1, 含自环)
            seq_adj: (B, N, N) 序列邻接指示 (0/1)
            mask:    (B, N) 有效节点掩码 (1=真实残基)

        坐标是否随消息传递更新由构造参数 update_coord 决定
        (消融 no_coord: 各层距离固定基于初始 3D 坐标)。
        """
        # 两两距离 (数值稳定: clamp)
        diff = x[:, :, None, :] - x[:, None, :, :]          # (B, N, N, 3)
        dist = diff.norm(dim=-1, keepdim=True)               # (B, N, N, 1)
        dist = torch.clamp(dist, min=0.0, max=100.0) / 10.0  # 以 10Å 为尺度归一

        # ---- 边消息 ----
        hi = h[:, :, None, :].expand(-1, -1, h.size(1), -1)  # (B, N, N, H)
        hj = h[:, None, :, :].expand(-1, h.size(1), -1, -1)  # (B, N, N, H)
        edge_in = torch.cat([hi, hj, dist, seq_adj.unsqueeze(-1)], dim=-1)
        m = self.edge_mlp(edge_in)                            # (B, N, N, H)
        m = m * adj.unsqueeze(-1)                             # 邻接掩码

        # ---- 坐标更新 (等变): x_i' = x_i + (1/D_i) * Σ_j w_ij * (x_i - x_j) ----
        # 注: diff = x_i - x_j 且 w_ij = sigmoid(.) > 0, 故原始坐标逐层向外扩张 (斥力形式);
        # 结构信息体现在各向异性 (w_ij -> 0 的接触对扩张更慢), 需经 R_g 归一化后分析。
        denom = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)       # (B, N, 1)
        if self.update_coord:
            w = torch.sigmoid(self.coord_mlp(m)) * adj.unsqueeze(-1)   # (B, N, N, 1)
            coord_msg = (w * diff).sum(dim=2)                          # (B, N, 3)
            x = x + coord_msg / denom

        # ---- 特征更新 (掩码均值聚合 + 残差) ----
        agg = (m / denom.unsqueeze(-1)).sum(dim=2)                 # (B, N, H)
        h = self.norm(h + self.node_mlp(torch.cat([h, agg], dim=-1)))

        # 填充位置强制归零
        h = h * mask.unsqueeze(-1)
        return h, x


class EGNNStack(nn.Module):
    """多层 E(n)-EGNN 堆叠。"""

    def __init__(self, hidden_dim=128, num_layers=4, dropout=0.1,
                 update_coord=True):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, dropout, update_coord=update_coord)
            for _ in range(num_layers)
        ])

    def forward(self, h, x, adj, seq_adj, mask):
        for layer in self.layers:
            h, x = layer(h, x, adj, seq_adj, mask)
        return h, x


# ---------------------------------------------------------------------------
# 双向交叉注意力融合 (序列 <-> 结构 co-attention)
# ---------------------------------------------------------------------------

class BidirectionalCrossAttention(nn.Module):
    """
    ESM 序列特征与 EGNN 空间特征的双向交叉注意力 + 门控残差融合:
        seq_Q -> graph_KV, graph_Q -> seq_KV, 门控加权求和。
    """

    def __init__(self, dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.attn_sg = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                             batch_first=True)   # seq <- graph
        self.attn_gs = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                             batch_first=True)   # graph <- seq
        self.norm_s = nn.LayerNorm(dim)
        self.norm_g = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Linear(dim, 2),      # 每个模态一个门
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, seq_feat, graph_feat, mask):
        """
        Args:
            seq_feat, graph_feat: (B, N, H)
            mask: (B, N), 1=有效
        Returns:
            fused: (B, N, H)
        """
        key_padding = ~mask.bool()   # MHA 约定: True 表示忽略

        # 序列 attend 结构 (捕获空间上邻近但序列上远离的基序关联)
        s2g, _ = self.attn_sg(self.norm_s(seq_feat), graph_feat, graph_feat,
                              key_padding_mask=key_padding)
        # 结构 attend 序列 (以语义信息修正空间表征)
        g2s, _ = self.attn_gs(self.norm_g(graph_feat), seq_feat, seq_feat,
                              key_padding_mask=key_padding)

        # 门控残差融合
        gate = torch.sigmoid(self.gate(torch.cat([seq_feat, s2g, g2s], dim=-1)))
        fused = (gate[..., 0:1] * (seq_feat + s2g) +
                 gate[..., 1:2] * (graph_feat + g2s))
        fused = fused * mask.unsqueeze(-1)

        # FFN 残差
        fused = fused + self.ffn(fused)
        return fused * mask.unsqueeze(-1)


# ---------------------------------------------------------------------------
# 完整模型
# ---------------------------------------------------------------------------

class NeuroPred3D(nn.Module):
    """
    NeuroPred-3D-EGNN: 序列-结构多模态神经肽预测模型。

    Args:
        node_feat_dim: 节点初始特征维度 (21 one-hot + 5 理化 = 26)
        esm_dim: ESM 表征维度 (由缓存自动决定, 如 t6_8M=320, t33=1280)
        hidden_dim: 主干隐藏维度
        egnn_layers: EGNN 层数
        heads: 交叉注意力头数
        dropout: Dropout 比例

    消融开关 (默认全开 = 完整模型; 关闭时对应模块整组不创建,
    state_dict 中不含其参数, 保证是"干净消融"):
        use_egnn:         False = no_3d, 去掉 3D-EGNN 空间分支, 仅 ESM 序列
        use_esm:          False = no_esm, 去掉 ESM 语义, 仅 one-hot/理化 + 3D 图
        use_cross_attn:   False = no_attn, 双向交叉注意力换等权平均融合
        use_coord_update: False = no_coord, EGNN 不做等变坐标更新 (静态坐标)
    """

    def __init__(self, node_feat_dim=NODE_FEAT_DIM, esm_dim=1280, hidden_dim=128,
                 egnn_layers=4, heads=4, dropout=0.2,
                 use_egnn=True, use_esm=True, use_cross_attn=True,
                 use_coord_update=True):
        super().__init__()
        # 交叉注意力仅在两个分支都存在时才可能启用
        use_fusion_attn = bool(use_egnn and use_esm and use_cross_attn)
        self.use_egnn = bool(use_egnn)
        self.use_esm = bool(use_esm)
        self.use_fusion_attn = use_fusion_attn
        self.use_coord_update = bool(use_coord_update)
        self.config = dict(node_feat_dim=node_feat_dim, esm_dim=esm_dim,
                           hidden_dim=hidden_dim, egnn_layers=egnn_layers,
                           heads=heads, dropout=dropout,
                           use_egnn=self.use_egnn, use_esm=self.use_esm,
                           use_cross_attn=bool(use_cross_attn),
                           use_coord_update=self.use_coord_update)

        # 节点初始化: 理化特征 (+ ESM 语义投影) 注入图节点
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        # ESM 序列语义投影 (no_esm 时整组不创建)
        if self.use_esm:
            self.esm_proj = nn.Sequential(
                nn.Linear(esm_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )

        # 3D 空间分支 (no_3d 时整组不创建)
        if self.use_egnn:
            self.egnn = EGNNStack(hidden_dim, egnn_layers, dropout,
                                  update_coord=self.use_coord_update)

        # 多模态融合: 完整模型用双向交叉注意力;
        # 任一消融变体改用与 fusion.ffn 同构的独立 FFN 做融合后变换,
        # 保证消融只移除目标组件、不额外改变网络深度。
        if use_fusion_attn:
            self.fusion = BidirectionalCrossAttention(hidden_dim, heads, dropout)
        else:
            self.post_fuse_ffn = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

        # 分类头 (掩码均值 + 最大值池化; dropout 与主干统一, 抑制过拟合)
        self.clf = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for module in [self.clf]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def masked_pool(self, h, mask):
        """掩码均值 + 最大值池化 -> (B, 2H)。"""
        m = mask.unsqueeze(-1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        neg = (~mask.bool()).float().unsqueeze(-1) * 1e9   # 填充位 -inf 量级
        mx = (h - neg).max(dim=1).values
        return torch.cat([mean, mx], dim=-1)

    def forward(self, node_feats, coords, adj, seq_adj, esm_emb, mask):
        """
        Args:
            node_feats: (B, N, 26)  one-hot + 理化性质
            coords:     (B, N, 3)   Cα 坐标 (EGNN 中会被等变更新)
            adj:        (B, N, N)   空间邻接
            seq_adj:    (B, N, N)   序列邻接指示
            esm_emb:    (B, N, D)   冻结 ESM-2 表征 (预计算)
            mask:       (B, N)      有效残基掩码
        Returns:
            (B, 2) softmax 概率

        注: 消融变体下 esm_emb/coords 等输入仍照常传入 (数据管线不变),
            只是对应分支模块不存在、前向不使用, 确保开关在模型内部统一门控。
        """
        m = mask.unsqueeze(-1)

        # ---- 节点初始化 / 序列分支 ----
        h = self.node_proj(node_feats)                       # (B, N, H)
        if self.use_esm:
            seq_feat = self.esm_proj(esm_emb)               # (B, N, H)
            h = h + seq_feat

        # ---- 3D 空间分支 ----
        if self.use_egnn:
            graph_feat, _ = self.egnn(h, coords, adj, seq_adj, mask)

        # ---- 融合 / 单分支 ----
        if self.use_fusion_attn:
            # 完整模型: 双向交叉注意力 (内含门控残差 + FFN)
            fused = self.fusion(seq_feat, graph_feat, mask)
        elif self.use_egnn and self.use_esm:
            # no_attn: 移除双向交叉注意力与门控, 两模态等权平均
            fused = 0.5 * (seq_feat + graph_feat) * m
            fused = fused + self.post_fuse_ffn(fused)
        elif self.use_esm:
            # no_3d: 仅 ESM 序列分支
            fused = seq_feat * m
            fused = fused + self.post_fuse_ffn(fused)
        else:
            # no_esm: 仅 3D 结构图分支
            fused = graph_feat * m
            fused = fused + self.post_fuse_ffn(fused)
        fused = fused * m                                   # (B, N, H)

        pooled = self.masked_pool(fused, mask)              # (B, 2H)
        logits = self.clf(pooled)
        return F.softmax(logits, dim=-1)


NeuroEGNN = NeuroPred3D


if __name__ == '__main__':
    # 模型形状自检: 完整模型 + 4 个消融变体
    B, N, D = 4, 50, 1280
    mask = torch.zeros(B, N)
    mask[:, :12] = 1.0
    variants = {
        'full':     dict(),
        'no_3d':    dict(use_egnn=False),
        'no_esm':   dict(use_esm=False),
        'no_attn':  dict(use_cross_attn=False),
        'no_coord': dict(use_coord_update=False),
    }
    for name, kw in variants.items():
        model = NeuroPred3D(esm_dim=D, **kw)
        out = model(
            torch.randn(B, N, 26), torch.randn(B, N, 3),
            (torch.rand(B, N, N) > 0.8).float() + torch.eye(N).unsqueeze(0),
            torch.eye(N).unsqueeze(0).expand(B, -1, -1),
            torch.randn(B, N, D), mask,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert out.shape == (B, 2), f'{name} 输出形状错误: {out.shape}'
        assert torch.allclose(out.sum(-1), torch.ones(B), atol=1e-4), 'softmax 概率和不为 1'
        print(f'{name:9s} 输出形状: {tuple(out.shape)} | 可训练参数量: {n_params / 1e6:.3f} M')
    print('全部 5 个变体前向自检通过。')
