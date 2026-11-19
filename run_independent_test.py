# -*- coding: utf-8 -*-
"""Out-of-fold K-fold evaluation and external test runner with sequence-overlap checks."""

import os
import sys
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from test import test
from train import full_metrics, format_metrics_text
from dataset import resolve_data_path

METRIC_COLS = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity',
               'MCC', 'AUC', 'F1-Score']
COUNT_COLS = ['TP', 'TN', 'FP', 'FN']


def parse_args():
    parser = argparse.ArgumentParser(description='Out-of-fold and external evaluation for K-fold NeuroEGNN checkpoints')
    parser.add_argument('--model_dir', type=str,
                        default=os.path.join('results', '5fold', 'results_5fold', 'train'),
                        help='Directory containing fold_<i>/best_model.pth checkpoints')
    parser.add_argument('--eval_mode', type=str, default='fold_test',
                        choices=['independent', 'fold_test'],
                        help='fold_test evaluates each fold on testing_v2_fold{i+1}.csv; independent evaluates an external CSV')
    parser.add_argument('--test_csv', type=str, default='testing_v2.csv',
                        help='External test CSV used when eval_mode=independent')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join('results', 'independent_test'))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--check_leakage', action='store_true', default=True,
                        help='Check exact sequence overlap against fold train/validation splits')
    return parser.parse_args()


def find_fold_models(model_dir):
    folds = []
    for i in range(10):
        fold_dir = os.path.join(model_dir, f'fold_{i}')
        if not os.path.isdir(fold_dir):
            continue
        for name in ('best_model.pth', 'best_model'):
            p = os.path.join(fold_dir, name)
            if os.path.isfile(p):
                folds.append((i, p))
                break
    return folds


def check_seq_leakage(test_csv, fold_idx, data_dir=SCRIPT_DIR):
    test_path = resolve_data_path(test_csv)
    test_df = pd.read_csv(test_path)
    test_seq_col = 'Seq' if 'Seq' in test_df.columns else 'Sequence'
    test_seqs = set(test_df[test_seq_col].astype(str).str.strip())

    train_file = os.path.join(data_dir, f'training_v2_fold{fold_idx + 1}.csv')
    val_file = os.path.join(data_dir, f'validation_v2_fold{fold_idx + 1}.csv')

    train_seqs = set()
    for fpath in (train_file, val_file):
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            scol = 'Seq' if 'Seq' in df.columns else 'Sequence'
            train_seqs.update(df[scol].astype(str).str.strip())

    if not train_seqs:
        return None, 0, 0.0

    overlap = test_seqs.intersection(train_seqs)
    overlap_pct = (len(overlap) / len(test_seqs)) * 100 if test_seqs else 0.0
    return overlap, len(overlap), overlap_pct


def main():
    args = parse_args()
    root = os.path.join(SCRIPT_DIR, args.out_dir)
    os.makedirs(root, exist_ok=True)

    folds = find_fold_models(os.path.join(SCRIPT_DIR, args.model_dir))
    if not folds:
        print(f"Error: no fold_<i>/best_model.pth checkpoints found under {args.model_dir}")
        sys.exit(1)

    print(f"[Eval] mode={args.eval_mode} | folds={len(folds)}")

    rows = []
    clean_rows = []
    all_oof_preds = []

    for fold_idx, model_path in folds:
        fold_dir = os.path.join(root, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)
        pred_csv = os.path.join(fold_dir, 'test_predictions.csv')
        metrics_txt = os.path.join(fold_dir, 'metrics.txt')

        current_test_csv = f'testing_v2_fold{fold_idx + 1}.csv' if args.eval_mode == 'fold_test' else args.test_csv

        overlap_seqs, n_overlap, overlap_pct = None, 0, 0.0
        if args.check_leakage and args.eval_mode == 'independent':
            overlap_seqs, n_overlap, overlap_pct = check_seq_leakage(
                current_test_csv, fold_idx, SCRIPT_DIR)
            if n_overlap > 0:
                print(f"[Warning] fold_{fold_idx}: {n_overlap} test sequences ({overlap_pct:.2f}%) "
                      f"overlap with fold training/validation splits.")

        metrics, _ = test(test_csv=current_test_csv,
                          model_path=model_path,
                          batch_size=args.batch_size,
                          output_csv=pred_csv,
                          device=args.device,
                          limit=args.limit)

        preds_df = pd.read_csv(pred_csv)
        preds_df['fold'] = fold_idx
        all_oof_preds.append(preds_df)

        clean_metrics = None
        if overlap_seqs and 0 < len(overlap_seqs) < len(preds_df):
            seq_col = preds_df.columns[0]
            clean_df = preds_df[~preds_df[seq_col].astype(str).str.strip().isin(overlap_seqs)]
            if len(clean_df) > 0:
                clean_metrics = full_metrics(clean_df['True_Label'].values,
                                             clean_df['Predicted_Label'].values,
                                             clean_df['Prediction_Probability'].values)
                clean_df.to_csv(os.path.join(fold_dir, 'clean_unseen_predictions.csv'), index=False)
                clean_rows.append({
                    'fold': fold_idx, 'unseen_n': len(clean_df),
                    **{c: clean_metrics[c] for c in METRIC_COLS + COUNT_COLS},
                })

        with open(metrics_txt, 'w', encoding='utf-8') as f:
            f.write(f'Evaluation Mode: {args.eval_mode} | fold_{fold_idx}\n')
            f.write(f'model: {model_path}\n')
            f.write(f'test: {current_test_csv}\n')
            if n_overlap > 0:
                f.write(f'Sequence overlap with train/val: {n_overlap} ({overlap_pct:.2f}%)\n')
            f.write(format_metrics_text(metrics, f'Fold {fold_idx} All Metrics'))
            if clean_metrics:
                f.write('\n' + format_metrics_text(
                    clean_metrics, f'Fold {fold_idx} Unseen Subset Metrics (n={len(clean_df)})'))

        rows.append({
            'fold': fold_idx, 'model': os.path.basename(model_path),
            'test_csv': current_test_csv,
            'leakage_n': n_overlap, 'leakage_pct': f'{overlap_pct:.1f}%',
            **{c: metrics[c] for c in METRIC_COLS + COUNT_COLS},
        })

    raw = pd.DataFrame(rows)
    raw_path = os.path.join(root, 'evaluation_raw.csv')
    raw.to_csv(raw_path, index=False, encoding='utf-8-sig')

    summary_rows = []
    for c in METRIC_COLS:
        vals = raw[c].astype(float)
        summary_rows.append({
            'metric': c,
            'mean': f'{vals.mean():.4f}',
            'std': f'{vals.std(ddof=1):.4f}' if len(vals) > 1 else '0.0000',
            'values': ' / '.join(f'{v:.4f}' for v in vals),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(root, 'evaluation_summary.csv')
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')

    lines = [
        'NeuroEGNN Evaluation Report',
        '=' * 75,
        f'Mode: {args.eval_mode} | Folds: {len(folds)}',
        f'Target: {args.test_csv if args.eval_mode == "independent" else "Per-fold held-out splits"}',
        '',
        f'{"Fold":<8s} {"Overlap":<10s} ' + '  '.join(f'{c:>11s}' for c in METRIC_COLS),
    ]
    for _, r in raw.iterrows():
        leak_str = f"{r['leakage_n']}({r['leakage_pct']})" if 'leakage_n' in r else '0'
        lines.append(f'fold_{int(r["fold"]):<4d} {leak_str:<10s} ' +
                     '  '.join(f'{r[c]:>11.4f}' for c in METRIC_COLS))

    lines.append('')
    lines.append(f'{"Mean":<8s} {"-":<10s} ' +
                 '  '.join(f'{raw[c].astype(float).mean():>11.4f}' for c in METRIC_COLS))
    lines.append(f'{"Std":<8s} {"-":<10s} ' +
                 '  '.join(f'{raw[c].astype(float).std(ddof=1):>11.4f}' for c in METRIC_COLS))

    if clean_rows:
        clean_df = pd.DataFrame(clean_rows)
        lines.extend(['', '=' * 75, 'Unseen Subset Summary (Excluding Overlapping Sequences)', '=' * 75])
        lines.append(f'{"Fold":<8s} {"Unseen_N":<10s} ' + '  '.join(f'{c:>11s}' for c in METRIC_COLS))
        for _, r in clean_df.iterrows():
            lines.append(f'fold_{int(r["fold"]):<4d} {int(r["unseen_n"]):<10d} ' +
                         '  '.join(f'{r[c]:>11.4f}' for c in METRIC_COLS))
        lines.append(f'{"Mean":<8s} {"-":<10s} ' +
                     '  '.join(f'{clean_df[c].astype(float).mean():>11.4f}' for c in METRIC_COLS))
        lines.append(f'{"Std":<8s} {"-":<10s} ' +
                     '  '.join(f'{clean_df[c].astype(float).std(ddof=1):>11.4f}' for c in METRIC_COLS))

    report_path = os.path.join(root, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    if args.eval_mode == 'fold_test' and all_oof_preds:
        oof_all = pd.concat(all_oof_preds, ignore_index=True)
        oof_path = os.path.join(root, 'oof_predictions_all.csv')
        oof_all.to_csv(oof_path, index=False)
        print(f"[OOF] Saved {len(oof_all)} out-of-fold predictions to {oof_path}")

    for _, r in summary.iterrows():
        print(f"  {r['metric']:15s}: {r['mean']} ± {r['std']}")


if __name__ == '__main__':
    main()
