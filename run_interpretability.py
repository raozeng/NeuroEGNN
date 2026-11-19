# -*- coding: utf-8 -*-
"""Extract and plot bidirectional sequence-structure cross-attention maps from NeuroEGNN."""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dataset import ensure_caches, get_dataloader_3d, SCRIPT_DIR as DS_DIR
from model import NeuroPred3D

CACHE_DIR = os.path.join(DS_DIR, 'cache')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not installed; saving attention arrays without plots.")


def parse_args():
    parser = argparse.ArgumentParser(description='Extract NeuroEGNN bidirectional cross-attention weights')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to a trained full-variant NeuroEGNN checkpoint')
    parser.add_argument('--test_csv', type=str, default='testing_v2.csv')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join('results', 'interpretability'))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=5)
    return parser.parse_args()


class AttentionExtractor:
    """Forward-hook collector for MultiheadAttention weights in BidirectionalCrossAttention."""

    def __init__(self, model):
        self.model = model
        self.attn_weights = {}
        self._hooks = []
        self._batch_idx = 0
        self._register_hooks()

    def _register_hooks(self):
        if not hasattr(self.model, 'fusion'):
            return
        fusion = self.model.fusion
        if not hasattr(fusion, 'attn_sg') or not hasattr(fusion, 'attn_gs'):
            return

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) == 2:
                    _, attn_weights = output
                    self.attn_weights.setdefault(self._batch_idx, {})[name] = attn_weights.detach().cpu()
            return hook

        self._hooks.append(fusion.attn_sg.register_forward_hook(make_hook('s2g')))
        self._hooks.append(fusion.attn_gs.register_forward_hook(make_hook('g2s')))

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
    labels = batch['labels'].to(device)
    inputs = {k: batch[k].to(device) for k in
              ('node_feats', 'coords', 'adj', 'seq_adj', 'esm_emb', 'mask')}

    extractor = AttentionExtractor(model)
    with torch.no_grad():
        outputs = model(**inputs)
    extractor.remove_hooks()

    probs = outputs[:, 1].cpu().numpy()
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

    attn = {}
    for _, weights in extractor.attn_weights.items():
        for k, v in weights.items():
            attn[k] = v

    return probs, preds, labels.cpu().numpy(), attn, batch


def plot_attention_heatmap(seq, attn_matrix, true_label, pred_label, prob,
                           save_path, title_prefix=''):
    if not HAS_MPL:
        return
    L = len(seq)
    if attn_matrix.ndim == 3:
        attn_matrix = attn_matrix.mean(axis=0)
    L = min(L, attn_matrix.shape[0])
    attn_matrix = attn_matrix[:L, :L]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
    ax.set_xlabel('Residue Index (Key)')
    ax.set_ylabel('Residue Index (Query)')
    ax.set_title(f'{title_prefix} Cross-Attention Matrix')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    residue_attn = attn_matrix.mean(axis=0)
    ax.bar(range(L), residue_attn, color='steelblue', alpha=0.8)
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title(f'{title_prefix} Residue-level Attention '
                 f'(True={true_label}, Pred={pred_label}, Prob={prob:.3f})')
    if L <= 60:
        ax.set_xticks(range(L))
        ax.set_xticklabels(list(seq[:L]), fontsize=6, rotation=90)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_residue_importance_comparison(seqs, labels, probs, residue_attns,
                                       save_path, top_k=5):
    if not HAS_MPL:
        return
    correct_mask = labels == (probs > 0.5).astype(int)
    correct_idx = np.where(correct_mask)[0]
    wrong_idx = np.where(~correct_mask)[0]

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

    model, config = _load_model(args.model_path, device)
    if not hasattr(model, 'fusion') or not hasattr(model.fusion, 'attn_sg'):
        print("Error: checkpoint does not contain BidirectionalCrossAttention; use a 'full' checkpoint.")
        sys.exit(1)

    esm_model = config.get('esm_model', 'esm2_t33_650M_UR50D')
    max_len = config.get('max_len', 50)
    knn = config.get('knn', 8)
    contact_dist = config.get('contact_dist', 10.0)

    test_npz = ensure_caches(args.test_csv, CACHE_DIR, esm_model=esm_model, max_len=max_len)
    test_loader, _ = get_dataloader_3d(
        args.test_csv, test_npz, batch_size=args.batch_size, shuffle=False,
        max_len=max_len, knn=knn, contact_dist=contact_dist, limit=args.limit)

    all_seqs, all_labels, all_probs, all_preds = [], [], [], []
    all_residue_attns = []
    sample_idx = 0

    for batch in test_loader:
        probs, preds, labels, attn, raw_batch = _forward_with_attention(model, batch, device)
        seqs = raw_batch['seqs']
        B = len(seqs)

        for i in range(B):
            seq = seqs[i]
            L = len(seq)
            label = int(labels[i])
            prob = float(probs[i])
            pred = int(preds[i])

            residue_attn = np.zeros(max_len)
            for key in ('s2g', 'g2s'):
                if key in attn:
                    a = attn[key]
                    if a.ndim == 3 and a.shape[0] == B:
                        a_i = a[i]
                    elif a.ndim == 3:
                        a_i = a.mean(axis=0)
                    else:
                        a_i = a
                    col_mean = a_i[:L, :L].mean(axis=0).numpy()
                    residue_attn[:col_mean.shape[0]] += col_mean
            residue_attn = residue_attn / max(len(attn), 1)

            all_seqs.append(seq)
            all_labels.append(label)
            all_probs.append(prob)
            all_preds.append(pred)
            all_residue_attns.append(residue_attn)

            np.savez(os.path.join(attn_dir, f'sample_{sample_idx:04d}.npz'),
                     seq=seq, label=label, prob=prob, pred=pred,
                     residue_attn=residue_attn,
                     **{k: v[i].numpy() if v.ndim >= 3 else v.numpy()
                        for k, v in attn.items()})
            sample_idx += 1

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    correct_mask = all_labels == all_preds
    correct_idx = np.where(correct_mask)[0]
    wrong_idx = np.where(~correct_mask)[0]
    top_k = args.top_k

    def _load_attn_matrix(idx):
        d = np.load(os.path.join(attn_dir, f'sample_{idx:04d}.npz'))
        for k in ('s2g', 'g2s'):
            if k in d:
                return d[k]
        raise KeyError(f'Missing s2g/g2s matrix in sample_{idx:04d}.npz')

    if len(correct_idx) > 0:
        top_correct = correct_idx[np.argsort(-all_probs[correct_idx])][:top_k]
        for rank, idx in enumerate(top_correct):
            plot_attention_heatmap(
                all_seqs[idx], _load_attn_matrix(idx),
                all_labels[idx], all_preds[idx], all_probs[idx],
                os.path.join(fig_dir, f'top_correct_{rank + 1}_heatmap.png'),
                title_prefix=f'Correct #{idx}')

    if len(wrong_idx) > 0:
        top_wrong = wrong_idx[np.argsort(all_probs[wrong_idx])][:top_k]
        for rank, idx in enumerate(top_wrong):
            plot_attention_heatmap(
                all_seqs[idx], _load_attn_matrix(idx),
                all_labels[idx], all_preds[idx], all_probs[idx],
                os.path.join(fig_dir, f'top_wrong_{rank + 1}_heatmap.png'),
                title_prefix=f'Wrong #{idx}')

    if len(correct_idx) > 0 and len(wrong_idx) > 0:
        plot_residue_importance_comparison(
            all_seqs, all_labels, all_probs, all_residue_attns,
            os.path.join(fig_dir, 'residue_attention_comparison.png'),
            top_k=top_k)

    report_lines = [
        'NeuroEGNN Cross-Attention Interpretability Report',
        '=' * 70,
        f'Checkpoint: {args.model_path}',
        f'Dataset: {args.test_csv} | Samples: {sample_idx}',
        f'Accuracy: {(all_labels == all_preds).mean():.4f}',
        f'Attention arrays: {attn_dir}/',
        f'Figures: {fig_dir}/',
    ]
    report_path = os.path.join(root, 'interpretability_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines) + '\n')

    print(f"[Interpretability] Extracted {sample_idx} samples -> {attn_dir} and {report_path}")


if __name__ == '__main__':
    main()
