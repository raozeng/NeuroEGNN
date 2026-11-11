# -*- coding: utf-8 -*-
"""
NeuroPred-3D-EGNN 可解释性分析

提取完整模型 (full) 的双向交叉注意力权重, 对测试集样本做可解释性可视化:
  1. 序列级注意力热力图: 每个残基对最终预测的贡献度 (s2g + g2s 注意力均值)
  2. 序列-结构交叉注意力矩阵: 哪些残基对在跨模态交互中最活跃
  3. 高置信正确/错误案例的注意力对比

产物 (默认 results/interpretability/):
    attention_weights/           每条序列的注意力权重 npz
    figures/                     PNG 可视化图
    interpretability_report.txt  摘要报告

用法:
    # 用独立划分 (hold-out) 训练的零泄漏 full 模型在测试集上做可解释性分析
    python run_interpretability.py \
        --model_path results/ablation/full/seed_500/train/best_model.pth \
        --test_csv testing_v2.csv

    # 只分析前 50 条
    python run_interpretability.py \
        --model_path results/ablation/full/seed_500/train/best_model.pth \
        --test_csv testing_v2.csv --limit 50
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dataset import ensure_caches, get_dataloader_3d, resolve_data_path, SCRIPT_DIR as DS_DIR
from model import NeuroPred3D

CACHE_DIR = os.path.join(DS_DIR, 'cache')

# 尝试导入 matplotlib; 如果不可用则跳过绘图
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print('警告: matplotlib 不可用, 将跳过可视化, 仅保存注意力权重数据。')


def parse_args():
    parser = argparse.ArgumentParser(
        description='NeuroPred-3D-EGNN 可解释性分析: 提取交叉注意力权重并可视化')
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型权重路径 (需为 full 变体)')
    parser.add_argument('--test_csv', type=str, default='testing_v2.csv')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join('results', 'interpretability'))
    parser.add_argument('--batch_size', type=int, default=32,
                        help='注意力提取用较小 batch (默认 32)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--limit', type=int, default=0,
                        help='只分析前 N 条 (0=全部)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='报告中展示注意力最高的 top-K 案例')
    return parser.parse_args()


class AttentionExtractor:
    """从 NeuroPred3D 模型中提取双向交叉注意力权重。"""

    def __init__(self, model):
        self.model = model
        self.attn_weights = {}  # {batch_idx: {'s2g': ..., 'g2s': ...}}
        self._hooks = []
        self._batch_idx = 0
        self._register_hooks()

    def _register_hooks(self):
        """在 MultiheadAttention 上注册 hook 来捕获注意力权重。"""
        if not hasattr(self.model, 'fusion'):
            return
        fusion = self.model.fusion
        if not hasattr(fusion, 'attn_sg') or not hasattr(fusion, 'attn_gs'):
            return

        def make_hook(name):
            def hook(module, input, output):
                # MultiheadAttention 输出: (attn_output, attn_output_weights)
                if isinstance(output, tuple) and len(output) == 2:
                    attn_out, attn_weights = output
                    # attn_weights: (B, N, N) 或 (B*num_heads, N, N)
                    # 注意: 两个 hook 共享同一 batch 槽位, 必须合并而非覆盖
                    self.attn_weights.setdefault(self._batch_idx, {})[name] = \
                        attn_weights.detach().cpu()
            return hook

        self._hooks.append(
            fusion.attn_sg.register_forward_hook(make_hook('s2g')))
        self._hooks.append(
            fusion.attn_gs.register_forward_hook(make_hook('g2s')))

    def reset(self):
        self.attn_weights = {}
        self._batch_idx = 0

    def next_batch(self):
        self._batch_idx += 1

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def _load_model(model_path, device):
    """加载模型并返回 (model, config)。"""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    model = NeuroPred3D(
        esm_dim=config.get('esm_dim', 1280),
        hidden_dim=config.get('hidden_dim', 128),
        egnn_layers=config.get('egnn_layers', 4),
        heads=config.get('heads', 4),
        dropout=config.get('dropout', 0.1),
        use_egnn=config.get('use_egnn', True),
        use_esm=config.get('use_esm', True),
        use_cross_attn=config.get('use_cross_attn', True),
        use_coord_update=config.get('use_coord_update', True))
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()
    return model, config


def _forward_with_attention(model, batch, device):
    """前向推理, 返回 (预测概率, 注意力权重 dict)。"""
    labels = batch['labels'].to(device)
    inputs = {k: batch[k].to(device) for k in
              ('node_feats', 'coords', 'adj', 'seq_adj', 'esm_emb', 'mask')}

    extractor = AttentionExtractor(model)
    with torch.no_grad():
        outputs = model(**inputs)
    extractor.remove_hooks()

    probs = outputs[:, 1].cpu().numpy()
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # 收集注意力权重 (取最后一个 batch 的)
    attn = {}
    for batch_idx, weights in extractor.attn_weights.items():
        for k, v in weights.items():
            attn[k] = v  # (B, N, N) 或 (B*heads, N, N)

    return probs, preds, labels.cpu().numpy(), attn, batch


def plot_attention_heatmap(seq, attn_matrix, true_label, pred_label, prob,
                           save_path, title_prefix=''):
    """绘制序列级注意力热力图 (残基 x 残基)。"""
    if not HAS_MPL:
        return
    L = len(seq)
    # 如果 attn 是多头平均后的 (N, N), 直接使用; 否则先平均
    if attn_matrix.ndim == 3:
        attn_matrix = attn_matrix.mean(axis=0)  # 平均多头
    # 序列长度可能超过 max_len (注意力矩阵只有 N=max_len), 截断对齐
    L = min(L, attn_matrix.shape[0])
    attn_matrix = attn_matrix[:L, :L]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: 注意力矩阵
    ax = axes[0]
    im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
    ax.set_xlabel('Residue Index (Key)')
    ax.set_ylabel('Residue Index (Query)')
    ax.set_title(f'{title_prefix} Cross-Attention Matrix')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 右: 每个残基的平均注意力 (列均值 = 该残基被关注的程度)
    ax = axes[1]
    residue_attn = attn_matrix.mean(axis=0)  # (N,) 列均值
    ax.bar(range(L), residue_attn, color='steelblue', alpha=0.8)
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title(f'{title_prefix} Residue-level Attention '
                 f'(True={true_label}, Pred={pred_label}, Prob={prob:.3f})')
    # 标注序列
    if L <= 60:
        ax.set_xticks(range(L))
        ax.set_xticklabels(list(seq[:L]), fontsize=6, rotation=90)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_residue_importance_comparison(seqs, labels, probs, residue_attns,
                                       save_path, top_k=5):
    """对比高置信正确与错误案例的残基注意力分布。"""
    if not HAS_MPL:
        return
    # 选 top-K 高置信正确 + top-K 高置信错误
    correct_mask = labels == (probs > 0.5).astype(int)
    correct_idx = np.where(correct_mask)[0]
    wrong_idx = np.where(~correct_mask)[0]

    # 按置信度排序
    correct_sorted = correct_idx[np.argsort(-probs[correct_idx])][:top_k]
    wrong_sorted = wrong_idx[np.argsort(probs[wrong_idx])][:top_k]

    fig, axes = plt.subplots(2, top_k, figsize=(4 * top_k, 8))
    if top_k == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(correct_sorted):
        ax = axes[0, col]
        attn_full = residue_attns[idx]
        L = min(len(seqs[idx]), len(attn_full))
        attn = attn_full[:L]
        ax.bar(range(L), attn, color='green', alpha=0.7)
        ax.set_title(f'Correct #{idx} (p={probs[idx]:.3f})', fontsize=9)
        ax.set_ylim(0, max(attn.max() * 1.2, 0.01))
        if L <= 40:
            ax.set_xticks(range(L))
            ax.set_xticklabels(list(seqs[idx][:L]), fontsize=5, rotation=90)

    for col, idx in enumerate(wrong_sorted):
        ax = axes[1, col]
        attn_full = residue_attns[idx]
        L = min(len(seqs[idx]), len(attn_full))
        attn = attn_full[:L]
        ax.bar(range(L), attn, color='red', alpha=0.7)
        ax.set_title(f'Wrong #{idx} (p={probs[idx]:.3f})', fontsize=9)
        ax.set_ylim(0, max(attn.max() * 1.2, 0.01))
        if L <= 40:
            ax.set_xticks(range(L))
            ax.set_xticklabels(list(seqs[idx][:L]), fontsize=5, rotation=90)

    axes[0, 0].set_ylabel('Correct', fontsize=11)
    axes[1, 0].set_ylabel('Wrong', fontsize=11)
    plt.suptitle('Residue-level Attention: Correct vs Wrong Predictions',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    root = os.path.join(SCRIPT_DIR, args.out_dir)
    attn_dir = os.path.join(root, 'attention_weights')
    fig_dir = os.path.join(root, 'figures')
    os.makedirs(attn_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    print('=' * 64)
    print(f'  可解释性分析 | 设备: {device}')
    print(f'  模型: {args.model_path}')
    print(f'  测试集: {args.test_csv}')
    print('=' * 64)

    # ---- 加载模型与数据 ----
    model, config = _load_model(args.model_path, device)
    if not hasattr(model, 'fusion') or not hasattr(model.fusion, 'attn_sg'):
        print('错误: 该模型不含交叉注意力模块 (可能是消融变体)。'
              '请使用 full 变体的权重。')
        sys.exit(1)

    esm_model = config.get('esm_model', 'esm2_t33_650M_UR50D')
    max_len = config.get('max_len', 50)
    knn = config.get('knn', 8)
    contact_dist = config.get('contact_dist', 10.0)

    test_npz = ensure_caches(args.test_csv, CACHE_DIR, esm_model=esm_model,
                             max_len=max_len)
    test_loader, test_dataset = get_dataloader_3d(
        args.test_csv, test_npz, batch_size=args.batch_size, shuffle=False,
        max_len=max_len, knn=knn, contact_dist=contact_dist, limit=args.limit)

    # ---- 逐 batch 提取注意力并保存 ----
    all_seqs, all_labels, all_probs, all_preds = [], [], [], []
    all_residue_attns = []  # 每条序列的残基级注意力 (L,)
    sample_idx = 0

    for batch in test_loader:
        probs, preds, labels, attn, raw_batch = \
            _forward_with_attention(model, batch, device)

        seqs = raw_batch['seqs']
        masks = raw_batch['mask'].numpy()

        for i in range(len(seqs)):
            seq = seqs[i]
            L = len(seq)
            label = int(labels[i])
            prob = float(probs[i])
            pred = int(preds[i])

            # 计算残基级注意力: s2g 和 g2s 的列均值 (被关注程度)
            residue_attn = np.zeros(max_len)
            B = len(seqs)  # 实际 batch 大小 (最后一个 batch 可能不足 batch_size)
            for key in ('s2g', 'g2s'):
                if key in attn:
                    a = attn[key]  # (B, N, N) 或 (B*heads, N, N)
                    if a.ndim == 3 and a.shape[0] == B:
                        a_i = a[i]  # (N, N) 当前样本的注意力矩阵
                    elif a.ndim == 3:
                        a_i = a.mean(axis=0)  # 多头未按 batch 拆分时退化为平均
                    else:
                        a_i = a  # 已经是 (N, N)
                    # 列均值 = 每个残基被其他残基关注的平均程度
                    # 短序列 (L < max_len) 的列均值只有 L 维, 累加进前 L 位
                    col_mean = a_i[:L, :L].mean(axis=0).numpy()
                    residue_attn[:col_mean.shape[0]] += col_mean

            residue_attn = residue_attn / max(len(attn), 1)  # 平均

            all_seqs.append(seq)
            all_labels.append(label)
            all_probs.append(prob)
            all_preds.append(pred)
            all_residue_attns.append(residue_attn)

            # 保存单条序列的注意力权重
            np.savez(os.path.join(attn_dir, f'sample_{sample_idx:04d}.npz'),
                     seq=seq, label=label, prob=prob, pred=pred,
                     residue_attn=residue_attn,
                     **{k: v[i].numpy() if v.ndim >= 3 else v.numpy()
                        for k, v in attn.items()})

            sample_idx += 1

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    print(f'共提取 {sample_idx} 条序列的注意力权重')

    # ---- 可视化 ----
    # 1. 高置信正确案例 top-K
    correct_mask = all_labels == all_preds
    correct_idx = np.where(correct_mask)[0]
    wrong_idx = np.where(~correct_mask)[0]

    top_k = args.top_k

    def _load_attn_matrix(idx):
        """加载单样本注意力矩阵; 兼容旧 npz 缺失 s2g 的情况 (回退 g2s)。"""
        d = np.load(os.path.join(attn_dir, f'sample_{idx:04d}.npz'))
        for k in ('s2g', 'g2s'):
            if k in d:
                return d[k]
        raise KeyError(f'sample_{idx:04d}.npz 中不含 s2g/g2s 注意力矩阵')

    # 高置信正确
    if len(correct_idx) > 0:
        top_correct = correct_idx[np.argsort(-all_probs[correct_idx])][:top_k]
        for rank, idx in enumerate(top_correct):
            plot_attention_heatmap(
                all_seqs[idx],
                _load_attn_matrix(idx),
                all_labels[idx], all_preds[idx], all_probs[idx],
                os.path.join(fig_dir, f'top_correct_{rank+1}_heatmap.png'),
                title_prefix=f'Correct #{idx}')

    # 高置信错误
    if len(wrong_idx) > 0:
        top_wrong = wrong_idx[np.argsort(all_probs[wrong_idx])][:top_k]
        for rank, idx in enumerate(top_wrong):
            plot_attention_heatmap(
                all_seqs[idx],
                _load_attn_matrix(idx),
                all_labels[idx], all_preds[idx], all_probs[idx],
                os.path.join(fig_dir, f'top_wrong_{rank+1}_heatmap.png'),
                title_prefix=f'Wrong #{idx}')

    # 2. 残基注意力对比图
    if len(correct_idx) > 0 and len(wrong_idx) > 0:
        plot_residue_importance_comparison(
            all_seqs, all_labels, all_probs, all_residue_attns,
            os.path.join(fig_dir, 'residue_attention_comparison.png'),
            top_k=top_k)

    # ---- 汇总报告 ----
    report_lines = [
        'NeuroPred-3D-EGNN 可解释性分析报告',
        '=' * 70,
        f'模型: {args.model_path}',
        f'测试集: {args.test_csv} | 样本数: {sample_idx}',
        f'准确率: {(all_labels == all_preds).mean():.4f}',
        '',
        f'注意力权重保存在: {attn_dir}/',
        f'可视化图保存在: {fig_dir}/',
        '',
        '文件说明:',
        '  sample_XXXX.npz  - 每条序列的注意力权重 (s2g/g2s) + 残基级注意力',
        '  top_correct_*_heatmap.png - 高置信正确案例的注意力热力图',
        '  top_wrong_*_heatmap.png   - 高置信错误案例的注意力热力图',
        '  residue_attention_comparison.png - 正确/错误案例残基注意力对比',
    ]
    report_path = os.path.join(root, 'interpretability_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines) + '\n')

    print('\n' + '=' * 64)
    print('可解释性分析完成')
    print('=' * 64)
    print(f'注意力权重: {attn_dir}/ ({sample_idx} 个 npz)')
    print(f'可视化图  : {fig_dir}/')
    print(f'报告      : {report_path}')


if __name__ == '__main__':
    main()
