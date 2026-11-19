# -*- coding: utf-8 -*-
"""Training routine for NeuroEGNN with label-smoothed BCE, cosine scheduling, and validation AUC early stopping."""

import os
import random
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix,
                             matthews_corrcoef, f1_score)

from dataset import (ensure_caches, get_dataloader_3d, resolve_data_path,
                     SCRIPT_DIR)
from model import NeuroPred3D, ABLATION_PRESETS

warnings.filterwarnings('ignore')

torch.set_num_threads(4)
if hasattr(torch, 'set_num_interop_threads'):
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass

CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')


def set_seed(seed=500):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(labels, preds, probs):
    labels, preds, probs = np.array(labels), np.array(preds), np.array(probs)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5
    mcc = matthews_corrcoef(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    return dict(acc=acc, sensitivity=sensitivity, specificity=specificity,
                auc=auc, mcc=mcc, f1=f1)


def full_metrics(labels, preds, probs):
    labels, preds, probs = np.array(labels), np.array(preds), np.array(probs)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5
    return {
        'Accuracy': accuracy_score(labels, preds),
        'Precision': precision_score(labels, preds, zero_division=0),
        'Sensitivity': recall_score(labels, preds, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'MCC': matthews_corrcoef(labels, preds),
        'AUC': auc,
        'F1-Score': f1_score(labels, preds, zero_division=0),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }


def format_metrics_text(metrics, title):
    lines = [
        '-' * 49, title, '-' * 49,
        f"Accuracy                      : {metrics['Accuracy']:.4f}",
        f"Precision                     : {metrics['Precision']:.4f}",
        f"Sensitivity (Recall)          : {metrics['Sensitivity']:.4f}",
        f"Specificity                   : {metrics['Specificity']:.4f}",
        f"Matthews Corr Coef (MCC)      : {metrics['MCC']:.4f}",
        f"AUC                           : {metrics['AUC']:.4f}",
        f"F1-Score                      : {metrics['F1-Score']:.4f}",
        f"Confusion Matrix (TP/TN/FP/FN): "
        f"{metrics['TP']} / {metrics['TN']} / {metrics['FP']} / {metrics['FN']}",
        '-' * 49,
    ]
    return '\n'.join(lines) + '\n'


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    all_seqs, all_labels, all_preds, all_probs = [], [], [], []
    for batch in dataloader:
        labels = batch['labels'].to(device)
        inputs = {k: batch[k].to(device) for k in
                  ('node_feats', 'coords', 'adj', 'seq_adj', 'esm_emb', 'mask')}
        outputs = model(**inputs)
        all_seqs.extend(batch['seqs'])
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_probs.extend(outputs[:, 1].cpu().numpy())
    return all_seqs, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def run_epoch(model, dataloader, criterion, optimizer, device, train_mode=True,
              label_smoothing=0.0):
    model.train() if train_mode else model.eval()
    total_loss, n_samples = 0.0, 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.set_grad_enabled(train_mode):
        for batch in dataloader:
            labels = batch['labels'].to(device)
            inputs = {k: batch[k].to(device) for k in
                      ('node_feats', 'coords', 'adj', 'seq_adj', 'esm_emb', 'mask')}

            outputs = model(**inputs)
            target = torch.nn.functional.one_hot(labels, num_classes=2).float()
            if label_smoothing > 0:
                target = target * (1.0 - label_smoothing) + label_smoothing / 2.0
            loss = criterion(outputs, target)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            n_samples += bs
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_probs.extend(outputs[:, 1].detach().cpu().numpy())

    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / max(n_samples, 1)
    return metrics


def train(train_csv='training_v2.csv',
          val_csv='validation_v2.csv',
          epochs=50,
          batch_size=128,
          lr=5e-4,
          weight_decay=5e-4,
          save_path=None,
          device=None,
          seed=500,
          esm_model='esm2_t33_650M_UR50D',
          max_len=50,
          knn=8,
          contact_dist=10.0,
          hidden_dim=128,
          egnn_layers=4,
          heads=4,
          dropout=0.2,
          patience=15,
          label_smoothing=0.05,
          use_api=False,
          api_limit=0,
          limit=0,
          out_dir=None,
          ablation='full'):
    """Train NeuroEGNN or an ablation variant and select the checkpoint maximizing validation AUC."""
    set_seed(seed)
    if ablation not in ABLATION_PRESETS:
        raise ValueError(f"Unknown ablation preset '{ablation}'. Available: {list(ABLATION_PRESETS)}")
    ablation_flags = ABLATION_PRESETS[ablation]
    device = torch.device(device) if device is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        latest_path = os.path.join(out_dir, 'latest_checkpoint.pth')
        log_path = os.path.join(out_dir, 'training_log.csv')
        val_metrics_path = os.path.join(out_dir, 'val_metrics.txt')
        val_pred_path = os.path.join(out_dir, 'val_predictions.csv')
        if save_path is None:
            save_path = os.path.join(out_dir, 'best_model.pth')
    else:
        latest_path = log_path = val_metrics_path = val_pred_path = None
        if save_path is None:
            save_path = os.path.join(SCRIPT_DIR, 'neuro3d_best_model.pth')

    print(f"[Train] NeuroEGNN ({ablation}) | device={device} | bs={batch_size} | lr={lr}")
    print(f"[Train] esm={esm_model} | egnn={egnn_layers}x{hidden_dim} | knn={knn} | cutoff={contact_dist}A")
    print(f"[Train] train={train_csv} | val={val_csv} | save={save_path}")

    train_npz = ensure_caches(train_csv, CACHE_DIR, esm_model=esm_model,
                              max_len=max_len, use_api=use_api, api_limit=api_limit)
    val_npz = ensure_caches(val_csv, CACHE_DIR, esm_model=esm_model, max_len=max_len)

    train_loader, _ = get_dataloader_3d(train_csv, train_npz, batch_size=batch_size,
                                        shuffle=True, max_len=max_len, knn=knn,
                                        contact_dist=contact_dist, limit=limit)
    val_loader, val_dataset = get_dataloader_3d(
        val_csv, val_npz, batch_size=batch_size,
        shuffle=False, max_len=max_len, knn=knn,
        contact_dist=contact_dist,
        limit=int(limit * 4) if limit else 0)

    model = NeuroPred3D(esm_dim=_read_esm_dim(train_npz), hidden_dim=hidden_dim,
                        egnn_layers=egnn_layers, heads=heads, dropout=dropout,
                        **ablation_flags).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] trainable_params={n_params / 1e6:.3f}M | train_batches={len(train_loader)} | val_batches={len(val_loader)}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_auc, best_acc, best_metrics, bad_epochs = 0.0, 0.0, {}, 0
    best_state = None
    history = []
    for epoch in range(epochs):
        train_m = run_epoch(model, train_loader, criterion, optimizer,
                            device, train_mode=True, label_smoothing=label_smoothing)
        val_m = run_epoch(model, val_loader, criterion, None,
                          device, train_mode=False)
        scheduler.step()

        saved_tag = ''
        if val_m['auc'] > best_auc + 1e-6:
            best_auc = val_m['auc']
            best_acc = val_m['acc']
            best_metrics = val_m.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'state_dict': best_state,
                        'config': {**model.config, 'ablation': ablation,
                                   'esm_model': esm_model,
                                   'max_len': max_len, 'knn': knn,
                                   'contact_dist': contact_dist}},
                       save_path)
            saved_tag = ' *'
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(f"Epoch {epoch:02d} | Train Loss: {train_m['loss']:.4f}, "
              f"Acc: {train_m['acc']:.4f}, Sens: {train_m['sensitivity']:.4f}, "
              f"Spec: {train_m['specificity']:.4f}, AUC: {train_m['auc']:.4f}, "
              f"MCC: {train_m['mcc']:.4f}, F1: {train_m['f1']:.4f}")
        print(f"Epoch {epoch:02d} | Valid Loss: {val_m['loss']:.4f}, "
              f"Acc: {val_m['acc']:.4f}, Sens: {val_m['sensitivity']:.4f}, "
              f"Spec: {val_m['specificity']:.4f}, AUC: {val_m['auc']:.4f}, "
              f"MCC: {val_m['mcc']:.4f}, F1: {val_m['f1']:.4f}{saved_tag}")

        history.append({
            'epoch': epoch,
            'train_loss': train_m['loss'], 'train_acc': train_m['acc'],
            'train_sensitivity': train_m['sensitivity'],
            'train_specificity': train_m['specificity'],
            'train_auc': train_m['auc'], 'train_mcc': train_m['mcc'],
            'train_f1': train_m['f1'],
            'val_loss': val_m['loss'], 'val_acc': val_m['acc'],
            'val_sensitivity': val_m['sensitivity'],
            'val_specificity': val_m['specificity'],
            'val_auc': val_m['auc'], 'val_mcc': val_m['mcc'],
            'val_f1': val_m['f1'],
            'best_val_auc': best_auc, 'saved_best': int(bool(saved_tag)),
        })
        if log_path is not None:
            pd.DataFrame(history).to_csv(log_path, index=False)
        if latest_path is not None:
            torch.save({
                'epoch': epoch,
                'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auc': best_auc,
                'config': {**model.config, 'ablation': ablation,
                           'esm_model': esm_model,
                           'max_len': max_len, 'knn': knn,
                           'contact_dist': contact_dist},
            }, latest_path)

        if bad_epochs >= patience:
            print(f"[EarlyStopping] Validation AUC did not improve for {patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if val_metrics_path is not None and val_pred_path is not None:
        val_seqs, val_labels, val_preds, val_probs = predict(model, val_loader, device)
        val_full = full_metrics(val_labels, val_preds, val_probs)
        raw_seqs = (val_dataset.dataset.raw_sequences
                    if isinstance(val_dataset, torch.utils.data.Subset)
                    else val_dataset.raw_sequences)
        pd.DataFrame({
            'Sequence': raw_seqs[:len(val_labels)],
            'True_Label': val_labels,
            'Predicted_Label': val_preds,
            'Prediction_Probability': val_probs,
        }).to_csv(val_pred_path, index=False)
        with open(val_metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"Validation metrics | {os.path.basename(str(val_csv))} | "
                    f"best selected by val AUC = {best_auc:.4f}\n")
            f.write(format_metrics_text(val_full, 'Validation Metrics'))

    print(f"[Train] Done | best_val_auc={best_auc:.4f} (val_acc={best_acc:.4f}) -> {save_path}")
    return {'best_auc': best_auc, 'best_acc': best_acc,
            'best_metrics': best_metrics, 'save_path': save_path,
            'n_params': n_params}


def _read_esm_dim(npz_path):
    with np.load(npz_path, allow_pickle=True) as data:
        return int(data['esm_dim'])


def parse_args():
    parser = argparse.ArgumentParser(description='Train NeuroEGNN or an ablation variant')
    parser.add_argument('--train_csv', type=str, default='training_v2.csv')
    parser.add_argument('--val_csv', type=str, default='validation_v2.csv',
                        help='Validation CSV used for early stopping and checkpoint selection')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--esm_model', type=str, default='esm2_t33_650M_UR50D',
                        help='Pretrained ESM-2 backbone name')
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--knn', type=int, default=8, help='k-nearest spatial neighbors')
    parser.add_argument('--contact_dist', type=float, default=10.0,
                        help='Spatial distance cutoff in Angstroms')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--egnn_layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                        help='Label smoothing factor (0 disables smoothing)')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=500)
    parser.add_argument('--use_api', action='store_true',
                        help='Query ESMFold online API for 3D coordinates when uncached')
    parser.add_argument('--api_limit', type=int, default=0,
                        help='Maximum number of ESMFold API requests (0 = unlimited)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Optional sample cap for quick debugging (0 = full dataset)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Directory for checkpoints, logs, and validation predictions')
    parser.add_argument('--ablation', type=str, default='full',
                        choices=list(ABLATION_PRESETS),
                        help='Model variant preset')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(train_csv=args.train_csv, val_csv=args.val_csv, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
          save_path=args.save_path, device=args.device, seed=args.seed,
          esm_model=args.esm_model, max_len=args.max_len, knn=args.knn,
          contact_dist=args.contact_dist, hidden_dim=args.hidden_dim,
          egnn_layers=args.egnn_layers, heads=args.heads, dropout=args.dropout,
          patience=args.patience, label_smoothing=args.label_smoothing,
          use_api=args.use_api, api_limit=args.api_limit, limit=args.limit,
          out_dir=args.out_dir, ablation=args.ablation)
