# -*- coding: utf-8 -*-
"""NeuroEGNN model definitions: dense E(3)-equivariant GNN and gated bidirectional cross-attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F

NODE_FEAT_DIM = 26

ABLATION_PRESETS = {
    'full':     dict(use_egnn=True,  use_esm=True,  use_cross_attn=True,  use_coord_update=True),
    'no_3d':    dict(use_egnn=False, use_esm=True,  use_cross_attn=True,  use_coord_update=True),
    'no_esm':   dict(use_egnn=True,  use_esm=False, use_cross_attn=True,  use_coord_update=True),
    'no_attn':  dict(use_egnn=True,  use_esm=True,  use_cross_attn=False, use_coord_update=True),
    'no_coord': dict(use_egnn=True,  use_esm=True,  use_cross_attn=True,  use_coord_update=False),
}


class EGNNLayer(nn.Module):
    """Dense-masked E(n)-equivariant graph layer (Satorras et al., ICML 2021)."""

    def __init__(self, hidden_dim=128, dropout=0.1, update_coord=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_coord = update_coord

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if update_coord:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.norm = nn.LayerNorm(hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, x, adj, seq_adj, mask):
        diff = x[:, :, None, :] - x[:, None, :, :]
        dist = diff.norm(dim=-1, keepdim=True)
        dist = torch.clamp(dist, min=0.0, max=100.0) / 10.0

        hi = h[:, :, None, :].expand(-1, -1, h.size(1), -1)
        hj = h[:, None, :, :].expand(-1, h.size(1), -1, -1)
        edge_in = torch.cat([hi, hj, dist, seq_adj.unsqueeze(-1)], dim=-1)
        m = self.edge_mlp(edge_in) * adj.unsqueeze(-1)

        # Positive sigmoid weights expand raw coordinates isotropically on average;
        # relative internal compaction is evaluated after Rg normalization.
        denom = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        if self.update_coord:
            w = torch.sigmoid(self.coord_mlp(m)) * adj.unsqueeze(-1)
            coord_msg = (w * diff).sum(dim=2)
            x = x + coord_msg / denom

        agg = (m / denom.unsqueeze(-1)).sum(dim=2)
        h = self.norm(h + self.node_mlp(torch.cat([h, agg], dim=-1)))
        h = h * mask.unsqueeze(-1)
        return h, x


class EGNNStack(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4, dropout=0.1, update_coord=True):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, dropout, update_coord=update_coord)
            for _ in range(num_layers)
        ])

    def forward(self, h, x, adj, seq_adj, mask):
        for layer in self.layers:
            h, x = layer(h, x, adj, seq_adj, mask)
        return h, x


class BidirectionalCrossAttention(nn.Module):
    """Reciprocal sequence-structure cross-attention with per-residue two-channel gating."""

    def __init__(self, dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.attn_sg = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_gs = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_s = nn.LayerNorm(dim)
        self.norm_g = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Linear(dim, 2),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, seq_feat, graph_feat, mask):
        key_padding = ~mask.bool()

        s2g, _ = self.attn_sg(self.norm_s(seq_feat), graph_feat, graph_feat,
                              key_padding_mask=key_padding)
        g2s, _ = self.attn_gs(self.norm_g(graph_feat), seq_feat, seq_feat,
                              key_padding_mask=key_padding)

        gate = torch.sigmoid(self.gate(torch.cat([seq_feat, s2g, g2s], dim=-1)))
        fused = (gate[..., 0:1] * (seq_feat + s2g) +
                 gate[..., 1:2] * (graph_feat + g2s))
        fused = fused * mask.unsqueeze(-1)
        fused = fused + self.ffn(fused)
        return fused * mask.unsqueeze(-1)


class NeuroPred3D(nn.Module):
    """Multimodal neuropeptide classifier combining frozen ESM-2 embeddings with a 3D E(3)-EGNN."""

    def __init__(self, node_feat_dim=NODE_FEAT_DIM, esm_dim=1280, hidden_dim=128,
                 egnn_layers=4, heads=4, dropout=0.2,
                 use_egnn=True, use_esm=True, use_cross_attn=True,
                 use_coord_update=True):
        super().__init__()
        use_fusion_attn = bool(use_egnn and use_esm and use_cross_attn)
        self.use_egnn = bool(use_egnn)
        self.use_esm = bool(use_esm)
        self.use_fusion_attn = use_fusion_attn
        self.use_coord_update = bool(use_coord_update)
        self.config = dict(
            node_feat_dim=node_feat_dim, esm_dim=esm_dim,
            hidden_dim=hidden_dim, egnn_layers=egnn_layers,
            heads=heads, dropout=dropout,
            use_egnn=self.use_egnn, use_esm=self.use_esm,
            use_cross_attn=bool(use_cross_attn),
            use_coord_update=self.use_coord_update,
        )

        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        if self.use_esm:
            self.esm_proj = nn.Sequential(
                nn.Linear(esm_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
        if self.use_egnn:
            self.egnn = EGNNStack(hidden_dim, egnn_layers, dropout,
                                  update_coord=self.use_coord_update)

        # Ablation variants without co-attention retain an isomorphic post-fusion FFN
        # so that post-fusion depth matches the full model.
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
        m = mask.unsqueeze(-1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        neg = (~mask.bool()).float().unsqueeze(-1) * 1e9
        mx = (h - neg).max(dim=1).values
        return torch.cat([mean, mx], dim=-1)

    def forward(self, node_feats, coords, adj, seq_adj, esm_emb, mask):
        m = mask.unsqueeze(-1)

        h = self.node_proj(node_feats)
        if self.use_esm:
            seq_feat = self.esm_proj(esm_emb)
            h = h + seq_feat

        if self.use_egnn:
            graph_feat, _ = self.egnn(h, coords, adj, seq_adj, mask)

        if self.use_fusion_attn:
            fused = self.fusion(seq_feat, graph_feat, mask)
        elif self.use_egnn and self.use_esm:
            fused = 0.5 * (seq_feat + graph_feat) * m
            fused = fused + self.post_fuse_ffn(fused)
        elif self.use_esm:
            fused = seq_feat * m
            fused = fused + self.post_fuse_ffn(fused)
        else:
            fused = graph_feat * m
            fused = fused + self.post_fuse_ffn(fused)
        fused = fused * m

        pooled = self.masked_pool(fused, mask)
        logits = self.clf(pooled)
        return F.softmax(logits, dim=-1)


NeuroEGNN = NeuroPred3D


if __name__ == '__main__':
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
        assert out.shape == (B, 2), f'Unexpected output shape for {name}: {out.shape}'
        assert torch.allclose(out.sum(-1), torch.ones(B), atol=1e-4)
        print(f'{name:9s} shape={tuple(out.shape)} | trainable_params={n_params / 1e6:.3f}M')
