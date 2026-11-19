# -*- coding: utf-8 -*-
"""Checkpoint evaluation and per-sequence prediction export for NeuroEGNN."""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, matthews_corrcoef,
                             roc_auc_score, f1_score)

from dataset import ensure_caches, get_dataloader_3d, resolve_data_path, SCRIPT_DIR
from model import NeuroPred3D

warnings.filterwarnings('ignore')

CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs, all_seqs = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels'].to(device)
            inputs = {k: batch[k].to(device) for k in
                      ('node_feats', 'coords', 'adj', 'seq_adj', 'esm_emb', 'mask')}
            outputs = model(**inputs)

            all_seqs.extend(batch['seqs'])
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_probs.extend(outputs[:, 1].cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    sens = recall_score(all_labels, all_preds, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    metrics = {
        'Accuracy': acc, 'Precision': prec, 'Sensitivity': sens,
        'Specificity': spec, 'MCC': mcc, 'AUC': auc, 'F1-Score': f1,
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }
    return metrics, all_seqs, all_labels, all_preds, all_probs


def test(test_csv='testing_v2.csv',
         model_path=None,
         batch_size=128,
         output_csv=None,
         device=None,
         limit=0):
    """Load a trained NeuroEGNN checkpoint, evaluate on test_csv, and save predictions."""
    device = torch.device(device) if device is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    if model_path is None:
        model_path = os.path.join(SCRIPT_DIR, 'neuro3d_best_model.pth')
    if output_csv is None:
        output_csv = os.path.join(SCRIPT_DIR, 'neuro3d_test_predictions.csv')

    if not os.path.exists(model_path):
        found = None
        for base in (os.getcwd(), SCRIPT_DIR, os.path.dirname(SCRIPT_DIR)):
            cand = os.path.join(base, os.path.basename(model_path))
            if os.path.exists(cand):
                found = cand
                break
        if found is None:
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        model_path = found

    print(f"[Eval] NeuroEGNN | device={device} | data={test_csv} | ckpt={model_path}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    esm_model = config.get('esm_model', 'esm2_t33_650M_UR50D')
    max_len = config.get('max_len', 50)
    knn = config.get('knn', 8)
    contact_dist = config.get('contact_dist', 10.0)

    test_npz = ensure_caches(test_csv, CACHE_DIR, esm_model=esm_model, max_len=max_len)
    test_loader, test_dataset = get_dataloader_3d(
        test_csv, test_npz, batch_size=batch_size, shuffle=False,
        max_len=max_len, knn=knn, contact_dist=contact_dist, limit=limit)

    model = NeuroPred3D(esm_dim=config.get('esm_dim', 1280),
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

    metrics, seqs, labels, preds, probs = evaluate(model, test_loader, device)

    print('-' * 49)
    print(f"Evaluation Summary ({config.get('ablation', 'full')})")
    print('-' * 49)
    print(f"Accuracy                      : {metrics['Accuracy']:.4f}")
    print(f"Precision                     : {metrics['Precision']:.4f}")
    print(f"Sensitivity (Recall)          : {metrics['Sensitivity']:.4f}")
    print(f"Specificity                   : {metrics['Specificity']:.4f}")
    print(f"Matthews Corr Coef (MCC)      : {metrics['MCC']:.4f}")
    print(f"AUC                           : {metrics['AUC']:.4f}")
    print(f"F1-Score                      : {metrics['F1-Score']:.4f}")
    print(f"Confusion Matrix (TP/TN/FP/FN): {metrics['TP']} / {metrics['TN']} / {metrics['FP']} / {metrics['FN']}")
    print('-' * 49)

    raw_seqs = test_dataset.dataset.raw_sequences if limit else test_dataset.raw_sequences
    results_df = pd.DataFrame({
        'Sequence': raw_seqs[:len(labels)],
        'True_Label': labels,
        'Predicted_Label': preds,
        'Prediction_Probability': probs,
    })
    results_df.to_csv(output_csv, index=False)
    print(f"[Eval] Saved {len(results_df)} predictions to {output_csv}")

    return metrics, results_df


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained NeuroEGNN checkpoint')
    parser.add_argument('--test_csv', type=str, default='testing_v2.csv')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    test(test_csv=args.test_csv, model_path=args.model_path,
         batch_size=args.batch_size, output_csv=args.output_csv,
         device=args.device, limit=args.limit)
