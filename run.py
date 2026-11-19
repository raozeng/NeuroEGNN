# -*- coding: utf-8 -*-
"""Unified CLI entry point for NeuroEGNN feature caching, training, evaluation, and K-fold CV."""

import os
import sys
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train import train
from test import test
from dataset import prepare_esm_cache, resolve_data_path


def parse_args():
    parser = argparse.ArgumentParser(description='NeuroEGNN CLI runner')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'train', 'test', 'prepare', 'cv'],
                        help='Execution mode: prepare / train / test / all / cv')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of cross-validation folds in cv mode')
    parser.add_argument('--cv_dir', type=str,
                        default=os.path.join('results', '5fold', 'results_5fold'),
                        help='Output directory for cross-validation runs')
    parser.add_argument('--train_csv', type=str, default='training_v2.csv')
    parser.add_argument('--val_csv', type=str, default='validation_v2.csv')
    parser.add_argument('--test_csv', type=str, default='testing_v2.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--esm_model', type=str, default='esm2_t33_650M_UR50D')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=500)
    parser.add_argument('--use_api', action='store_true',
                        help='Query ESMFold online API during feature preparation')
    parser.add_argument('--api_limit', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    print(f"[NeuroEGNN] mode={args.mode} | esm={args.esm_model} | device={args.device or 'auto'}")

    if args.mode == 'prepare':
        for csv_path in (args.train_csv, args.test_csv):
            prepare_esm_cache(resolve_data_path(csv_path), os.path.join(SCRIPT_DIR, 'cache'),
                              esm_model=args.esm_model, max_len=50,
                              use_api=args.use_api, api_limit=args.api_limit)
        return

    if args.mode == 'cv':
        import pandas as pd
        from train import format_metrics_text

        results_root = os.path.join(SCRIPT_DIR, args.cv_dir)
        train_root = os.path.join(results_root, 'train')
        test_root = os.path.join(results_root, 'test')
        os.makedirs(train_root, exist_ok=True)
        os.makedirs(test_root, exist_ok=True)

        rows = []
        # Output directories are 0-indexed (fold_0..fold_{K-1}), mapped to 1-indexed *_v2_fold{1..K}.csv splits.
        for fold in range(1, args.folds + 1):
            fold_idx = fold - 1
            train_csv = f'training_v2_fold{fold}.csv'
            val_csv = f'validation_v2_fold{fold}.csv'
            test_csv = f'testing_v2_fold{fold}.csv'
            fold_dir_name = f'fold_{fold_idx}'
            fold_train_dir = os.path.join(train_root, fold_dir_name)
            fold_test_dir = os.path.join(test_root, fold_dir_name)
            os.makedirs(fold_train_dir, exist_ok=True)
            os.makedirs(fold_test_dir, exist_ok=True)

            best_model = os.path.join(fold_train_dir, 'best_model.pth')
            test_pred_csv = os.path.join(fold_test_dir, 'test_predictions.csv')
            test_metrics_txt = os.path.join(fold_test_dir, 'metrics.txt')

            print(f"\n[CV] Fold {fold_idx}/{args.folds - 1} ({test_csv})")
            train(train_csv=train_csv, val_csv=val_csv, epochs=args.epochs,
                  batch_size=args.batch_size, lr=args.lr, save_path=best_model,
                  device=args.device, seed=args.seed, esm_model=args.esm_model,
                  use_api=args.use_api, api_limit=args.api_limit,
                  limit=args.limit, out_dir=fold_train_dir)

            metrics, _ = test(test_csv=test_csv, model_path=best_model,
                              batch_size=args.batch_size,
                              output_csv=test_pred_csv, device=args.device,
                              limit=args.limit)
            with open(test_metrics_txt, 'w', encoding='utf-8') as f:
                f.write(f"Fold {fold_idx} test metrics ({test_csv})\n"
                        f"train: {train_csv} | val: {val_csv} | test: {test_csv}\n")
                f.write(format_metrics_text(metrics, f'Test Metrics (fold_{fold_idx})'))

            rows.append({'Fold': fold_idx, 'train_csv': train_csv,
                         'val_csv': val_csv, 'test_csv': test_csv, **metrics})

        df = pd.DataFrame(rows)
        metric_cols = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity',
                       'MCC', 'AUC', 'F1-Score']
        count_cols = ['TP', 'TN', 'FP', 'FN']
        float_cols = [c for c in df.columns if df[c].dtype.kind == 'f']

        stat_row = {'Fold': 'Mean±Std', 'train_csv': '', 'val_csv': '', 'test_csv': ''}
        for c in float_cols:
            stat_row[c] = f"{df[c].mean():.4f}±{df[c].std(ddof=1):.4f}"
        for c in count_cols:
            if c in df.columns:
                stat_row[c] = int(df[c].sum())
        df_out = pd.concat([df, pd.DataFrame([stat_row])], ignore_index=True)
        results_path = os.path.join(results_root, 'cv_results.csv')
        df_out.to_csv(results_path, index=False)

        mean_std_path = os.path.join(results_root, 'mean_std.txt')
        with open(mean_std_path, 'w', encoding='utf-8') as f:
            f.write(f'{args.folds}-fold cross-validation (Mean ± Std)\n')
            f.write('=' * 50 + '\n')
            for c in metric_cols:
                f.write(f"{c:12s}: {df[c].mean():.4f} ± {df[c].std(ddof=1):.4f}\n")
            f.write('-' * 50 + '\n')
            f.write('Per-fold metrics:\n')
            f.write(df[['Fold'] + metric_cols].to_string(index=False))
            f.write('\n')

        print(f"\n[CV] Summary ({args.folds} folds):")
        print(df[['Fold'] + metric_cols + count_cols].to_string(index=False))
        for c in metric_cols:
            print(f"  {c:12s}: {df[c].mean():.4f} ± {df[c].std(ddof=1):.4f}")
        print(f"[CV] Saved {results_path} and {mean_std_path}")
        return

    if args.mode in ('all', 'train'):
        train(train_csv=args.train_csv, val_csv=args.val_csv,
              epochs=args.epochs, batch_size=args.batch_size,
              lr=args.lr, save_path=args.model_path,
              device=args.device, seed=args.seed,
              esm_model=args.esm_model, use_api=args.use_api,
              api_limit=args.api_limit, limit=args.limit)

    if args.mode in ('all', 'test'):
        test(test_csv=args.test_csv,
             model_path=args.model_path,
             batch_size=args.batch_size,
             output_csv=args.output_csv,
             device=args.device,
             limit=args.limit)

    print(f"[NeuroEGNN] Completed in {time.time() - start_time:.1f}s.")


if __name__ == '__main__':
    main()
