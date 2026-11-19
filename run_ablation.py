# -*- coding: utf-8 -*-
"""Multi-seed component ablation runner on the homology-isolated hold-out benchmark."""

import os
import sys
import time
import argparse
import traceback

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train import train, format_metrics_text, full_metrics
from test import test
from model import ABLATION_PRESETS

METRIC_COLS = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity',
               'MCC', 'AUC', 'F1-Score']
COUNT_COLS = ['TP', 'TN', 'FP', 'FN']


def parse_args():
    parser = argparse.ArgumentParser(description='Run multi-seed NeuroEGNN component ablations')
    parser.add_argument('--modes', nargs='+', default=list(ABLATION_PRESETS),
                        choices=list(ABLATION_PRESETS),
                        help='Ablation variants to run')
    parser.add_argument('--seeds', nargs='+', type=int, default=[500, 501, 502],
                        help='Random seeds per variant')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--esm_model', type=str, default='esm2_t33_650M_UR50D')
    parser.add_argument('--train_csv', type=str, default='training_v2.csv')
    parser.add_argument('--val_csv', type=str, default='validation_v2.csv')
    parser.add_argument('--test_csv', type=str, default='testing_v2.csv')
    parser.add_argument('--out_dir', type=str, default=os.path.join('results', 'ablation'))
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--force', action='store_true',
                        help='Re-run even when existing checkpoints and predictions are present')
    return parser.parse_args()


def run_one(variant, seed, args, root):
    unit_dir = os.path.join(root, variant, f'seed_{seed}')
    train_dir = os.path.join(unit_dir, 'train')
    test_dir = os.path.join(unit_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    best_model = os.path.join(train_dir, 'best_model.pth')
    test_pred_csv = os.path.join(test_dir, 'test_predictions.csv')
    test_metrics_txt = os.path.join(test_dir, 'metrics.txt')

    done = (os.path.exists(best_model) and
            os.path.exists(test_pred_csv) and
            os.path.exists(test_metrics_txt))
    if done and not args.force:
        print(f"[Skip] {variant}/seed_{seed} already exists (pass --force to retrain).")
        prev = pd.read_csv(test_pred_csv)
        metrics = full_metrics(prev['True_Label'].values,
                               prev['Predicted_Label'].values,
                               prev['Prediction_Probability'].values)
        ckpt = __import__('torch').load(best_model, map_location='cpu', weights_only=False)
        n_params = int(sum(v.numel() for v in ckpt['state_dict'].values()))
        return {'variant': variant, 'seed': seed, 'n_params': n_params,
                'status': 'cached', **metrics}

    print(f"\n[Ablation] {variant} | seed={seed} | flags={ABLATION_PRESETS[variant]}")
    t0 = time.time()

    train_result = train(
        train_csv=args.train_csv, val_csv=args.val_csv,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        save_path=best_model, device=args.device, seed=seed,
        esm_model=args.esm_model, limit=args.limit, out_dir=train_dir,
        ablation=variant)

    metrics, _ = test(test_csv=args.test_csv, model_path=best_model,
                      batch_size=args.batch_size, output_csv=test_pred_csv,
                      device=args.device, limit=args.limit)
    with open(test_metrics_txt, 'w', encoding='utf-8') as f:
        f.write(f'Ablation: {variant} | seed: {seed} | ESM: {args.esm_model}\n')
        f.write(f'train: {args.train_csv} | val: {args.val_csv} | test: {args.test_csv}\n')
        f.write(format_metrics_text(metrics, f'Test Metrics ({variant})'))

    print(f"[Ablation] Finished {variant}/seed_{seed} in {time.time() - t0:.0f}s (AUC={metrics['AUC']:.4f})")
    return {'variant': variant, 'seed': seed,
            'n_params': int(train_result['n_params']),
            'status': 'trained', **metrics}


def write_outputs(rows, args, root):
    raw_path = os.path.join(root, 'ablation_raw.csv')
    summary_path = os.path.join(root, 'ablation_summary.csv')
    report_path = os.path.join(root, 'ablation_report.txt')

    raw = pd.DataFrame(rows)
    raw = raw[['variant', 'seed', 'n_params', 'status'] + METRIC_COLS + COUNT_COLS]
    raw.to_csv(raw_path, index=False, encoding='utf-8-sig')

    summary_rows = []
    for variant in args.modes:
        sub = raw[raw['variant'] == variant]
        row = {'variant': variant, 'n_runs': len(sub),
               'n_params': int(sub['n_params'].iloc[0]) if len(sub) else 0}
        for c in METRIC_COLS:
            vals = sub[c].astype(float)
            row[f'{c}_mean'] = vals.mean()
            row[f'{c}_std'] = vals.std(ddof=1) if len(vals) > 1 else 0.0
            row[c] = (f'{vals.mean():.4f}±{vals.std(ddof=1):.4f}'
                      if len(vals) > 1 else f'{vals.mean():.4f}')
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')

    lines = [
        'NeuroEGNN Component Ablation Report',
        '=' * 90,
        f'Splits: train={args.train_csv} | val={args.val_csv} | test={args.test_csv}',
        f'ESM: {args.esm_model} | epochs: {args.epochs} | batch_size: {args.batch_size} | '
        f'lr: {args.lr} | seeds: {args.seeds}',
        '',
    ]
    header = f'{"Variant":10s} {"Params":>8s}  ' + '  '.join(f'{c:>11s}' for c in METRIC_COLS)
    lines.extend([header, '-' * len(header)])
    for _, r in summary.iterrows():
        lines.append(f'{r["variant"]:10s} {r["n_params"]/1e6:7.3f}M  ' +
                     '  '.join(f'{r[c]:>11s}' for c in METRIC_COLS))
    lines.append('')

    full = raw[raw['variant'] == 'full']
    if len(full):
        lines.append('Seed-paired difference relative to full model (delta = variant - full):')
        lines.append('-' * 90)
        lines.append(f'{"Variant":10s} ' + '  '.join(f'{("d_"+c):>11s}' for c in METRIC_COLS))
        for variant in args.modes:
            if variant == 'full':
                continue
            sub = raw[raw['variant'] == variant]
            merged = sub.merge(full, on='seed', suffixes=('', '_full'))
            if not len(merged):
                continue
            deltas = [f'{(merged[c] - merged[f"{c}_full"]).mean():+.4f}' for c in METRIC_COLS]
            lines.append(f'{variant:10s} ' + '  '.join(f'{d:>11s}' for d in deltas))
        lines.append('')

    lines.append('Per-seed metrics:')
    lines.append('-' * 90)
    show = raw[['variant', 'seed'] + METRIC_COLS].copy()
    lines.append(show.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n' + header)
    print('-' * len(header))
    for _, r in summary.iterrows():
        print(f'{r["variant"]:10s} {r["n_params"]/1e6:7.3f}M  ' +
              '  '.join(f'{r[c]:>11s}' for c in METRIC_COLS))
    print(f"[Ablation] Saved {raw_path}, {summary_path}, and {report_path}")


def main():
    args = parse_args()
    root = os.path.join(SCRIPT_DIR, args.out_dir)
    os.makedirs(root, exist_ok=True)

    rows = []
    t_start = time.time()
    for variant in args.modes:
        for seed in args.seeds:
            try:
                row = run_one(variant, seed, args, root)
                rows.append(row)
            except Exception as e:
                traceback.print_exc()
                rows.append({'variant': variant, 'seed': seed,
                             'n_params': 0, 'status': f'failed: {e}',
                             **{c: np.nan for c in METRIC_COLS},
                             **{c: np.nan for c in COUNT_COLS}})
            if rows:
                pd.DataFrame(rows).to_csv(
                    os.path.join(root, 'ablation_raw.csv'),
                    index=False, encoding='utf-8-sig')

    ok = [r for r in rows if not str(r['status']).startswith('failed')]
    if ok:
        write_outputs(rows, args, root)
    print(f"[Ablation] Completed {len(ok)}/{len(rows)} runs in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == '__main__':
    main()
