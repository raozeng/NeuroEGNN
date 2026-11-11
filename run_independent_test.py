# -*- coding: utf-8 -*-
"""
NeuroPred-3D-EGNN 独立测试运行器 (含数据泄漏防护与多模式评估)

支持两种评估模式:
    1. independent: 将五折模型在指定测试集上推理 (仅用于外部数据集)。
       NOTE: the 5-fold models were trained on folds of all_data.csv, which overlaps
       testing_v2.csv; hold-out results in the paper use results/ablation/full/seed_*/.
       - 自动执行【数据泄漏自检】: 检测测试集中是否混入了各折训练池中的序列;
       - 若检测到重叠, 会打印警告并分别报告【全量测试集】与【纯净未见子集 (Clean Unseen)】指标, 杜绝虚高评估。
    2. fold_test (默认): 五折交叉验证标准评估。
       - 每折模型自动评估自己对应的 held-out 测试集 (testing_v2_fold{i+1}.csv);
       - 理论保证 0 泄漏, 并合并 5 折预测生成完整的 Out-Of-Fold (OOF) 汇总评估。

用法:
    # 模式 1: 在独立测试集上评估 (自动泄漏自检)
    python run_independent_test.py --eval_mode independent --test_csv external.csv

    # 模式 2: 五折标准 Out-of-Fold 评估 (各折评估各折对应 test csv, 零泄漏)
    python run_independent_test.py --eval_mode fold_test
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from test import test, evaluate
from train import full_metrics, format_metrics_text, calculate_metrics
from dataset import resolve_data_path, get_dataloader_3d, ensure_caches

METRIC_COLS = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity',
               'MCC', 'AUC', 'F1-Score']
COUNT_COLS = ['TP', 'TN', 'FP', 'FN']


def parse_args():
    parser = argparse.ArgumentParser(
        description='五折模型在独立测试集/五折互斥集上的规范化评估')
    parser.add_argument('--model_dir', type=str,
                        default=os.path.join('results', '5fold',
                                             'results_5fold', 'train'),
                        help='五折训练产物根目录 (每折一个 fold_<i> 子目录)')
    parser.add_argument('--eval_mode', type=str, default='fold_test',
                        choices=['independent', 'fold_test'],
                        help='评估模式: independent (评估同一测试集) / fold_test (各折评估对应 testing_v2_fold{i}.csv)')
    parser.add_argument('--test_csv', type=str, default='testing_v2.csv',
                        help='independent 模式下使用的测试集路径')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join('results', 'independent_test'))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--check_leakage', action='store_true', default=True,
                        help='自动检测测试集与本折训练集/验证集的序列重叠与数据泄漏')
    return parser.parse_args()


def find_fold_models(model_dir):
    """扫描 model_dir, 返回 [(fold_idx, best_model_path), ...] 按折序排列。"""
    folds = []
    for i in range(10):  # 最多扫 10 折
        fold_dir = os.path.join(model_dir, f'fold_{i}')
        if not os.path.isdir(fold_dir):
            continue
        for name in ('best_model.pth', 'best_model'):
            p = os.path.join(fold_dir, name)
            if os.path.isfile(p):
                folds.append((i, p))
                break
            if os.path.isdir(p):
                print(f'警告: fold_{i} 的 {name} 是 PyTorch 分片存档目录, '
                      f'当前 torch.load 不支持目录加载, 已跳过。')
                continue
    return folds


def check_seq_leakage(test_csv, fold_idx, data_dir=SCRIPT_DIR):
    """
    检查指定测试集与该折训练集 (training_v2_fold{fold_idx+1}.csv)
    及验证集 (validation_v2_fold{fold_idx+1}.csv) 的序列重叠情况。
    """
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
        print(f'错误: 在 {args.model_dir} 下未找到任何 fold_<i>/best_model* 文件')
        sys.exit(1)

    print('=' * 68)
    print(f'  NeuroPred-3D-EGNN 评估运行器 | 模式: {args.eval_mode}')
    print(f'  模型数量: {len(folds)} 折 | 目标测试数据: '
          f'{"各折对应 testing_v2_fold{i}.csv" if args.eval_mode == "fold_test" else args.test_csv}')
    print('=' * 68)

    rows = []
    clean_rows = []
    all_oof_preds = []

    for fold_idx, model_path in folds:
        fold_dir = os.path.join(root, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)
        pred_csv = os.path.join(fold_dir, 'test_predictions.csv')
        metrics_txt = os.path.join(fold_dir, 'metrics.txt')

        # 确定本折使用的测试集
        if args.eval_mode == 'fold_test':
            current_test_csv = f'testing_v2_fold{fold_idx + 1}.csv'
        else:
            current_test_csv = args.test_csv

        print(f'\n>>> fold_{fold_idx}: 模型权重: {model_path}')
        print(f'    测试数据: {current_test_csv}')

        # 泄漏自检
        overlap_seqs, n_overlap, overlap_pct = None, 0, 0.0
        if args.check_leakage and args.eval_mode == 'independent':
            overlap_seqs, n_overlap, overlap_pct = check_seq_leakage(
                current_test_csv, fold_idx, SCRIPT_DIR)
            if n_overlap > 0:
                print(f'    [⚠️ 严重数据泄漏警告] 测试集中有 {n_overlap} 条序列 '
                      f'({overlap_pct:.2f}%) 曾出现在 fold_{fold_idx} 的训练集/验证集中!')

        metrics, _ = test(test_csv=current_test_csv,
                          model_path=model_path,
                          batch_size=args.batch_size,
                          output_csv=pred_csv,
                          device=args.device,
                          limit=args.limit)

        # 记录预测明细并支持纯净子集评估
        preds_df = pd.read_csv(pred_csv)
        preds_df['fold'] = fold_idx
        all_oof_preds.append(preds_df)

        clean_metrics = None
        if overlap_seqs and len(overlap_seqs) > 0 and len(overlap_seqs) < len(preds_df):
            seq_col = preds_df.columns[0]
            clean_df = preds_df[~preds_df[seq_col].astype(str).str.strip().isin(overlap_seqs)]
            if len(clean_df) > 0:
                clean_metrics = full_metrics(clean_df['True_Label'].values,
                                             clean_df['Predicted_Label'].values,
                                             clean_df['Prediction_Probability'].values)
                clean_csv = os.path.join(fold_dir, 'clean_unseen_predictions.csv')
                clean_df.to_csv(clean_csv, index=False)
                clean_row = {'fold': fold_idx, 'unseen_n': len(clean_df),
                             **{c: clean_metrics[c] for c in METRIC_COLS + COUNT_COLS}}
                clean_rows.append(clean_row)

        with open(metrics_txt, 'w', encoding='utf-8') as f:
            f.write(f'Evaluation Mode: {args.eval_mode} | fold_{fold_idx}\n')
            f.write(f'model: {model_path}\n')
            f.write(f'test: {current_test_csv}\n')
            if n_overlap > 0:
                f.write(f'Leakage Warning: {n_overlap} sequences ({overlap_pct:.2f}%) overlapped with train/val.\n')
            f.write(format_metrics_text(metrics, f'Fold {fold_idx} All Metrics'))
            if clean_metrics:
                f.write('\n' + format_metrics_text(clean_metrics, f'Fold {fold_idx} Clean Unseen Metrics (n={len(clean_df)})'))

        row = {'fold': fold_idx, 'model': os.path.basename(model_path),
               'test_csv': current_test_csv,
               'leakage_n': n_overlap, 'leakage_pct': f'{overlap_pct:.1f}%',
               **{c: metrics[c] for c in METRIC_COLS + COUNT_COLS}}
        rows.append(row)
        print(f'    fold_{fold_idx} 完成: AUC={metrics["AUC"]:.4f}, Acc={metrics["Accuracy"]:.4f}'
              + (f' (纯净未见样本 AUC: {clean_metrics["AUC"]:.4f})' if clean_metrics else ''))

    # ---- 汇总 ----
    raw = pd.DataFrame(rows)
    raw_path = os.path.join(root, 'evaluation_raw.csv')
    raw.to_csv(raw_path, index=False, encoding='utf-8-sig')

    def calc_summary(df, cols=METRIC_COLS):
        summary_rows = []
        for c in cols:
            vals = df[c].astype(float)
            summary_rows.append({
                'metric': c,
                'mean': f'{vals.mean():.4f}',
                'std': f'{vals.std(ddof=1):.4f}' if len(vals) > 1 else '0.0000',
                'values': ' / '.join(f'{v:.4f}' for v in vals),
            })
        return pd.DataFrame(summary_rows)

    summary = calc_summary(raw)
    summary_path = os.path.join(root, 'evaluation_summary.csv')
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')

    # ---- 文本报告 ----
    lines = ['NeuroPred-3D-EGNN 评估报告', '=' * 75,
             f'评估模式: {args.eval_mode} | 模型数: {len(folds)}',
             f'测试目标: {args.test_csv if args.eval_mode == "independent" else "各折独立测试集 (Fold Test)"}',
             '']

    has_leakage = raw['leakage_n'].sum() > 0 if 'leakage_n' in raw.columns else False
    if has_leakage:
        lines.append('⚠️ 注意: 检测到部分序列与训练集重叠 (详见下表 leakage 列)。')

    lines.append(f'{"Fold":<8s} {"Leakage":<10s} ' + '  '.join(f'{c:>11s}' for c in METRIC_COLS))
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
        lines.extend(['', '=' * 75, '纯净未见样本评估汇总 (Filtered Out Training-Seen Samples)', '=' * 75])
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

    # 如果是 fold_test 模式, 输出全量合并的 OOF 预测
    if args.eval_mode == 'fold_test' and all_oof_preds:
        oof_all = pd.concat(all_oof_preds, ignore_index=True)
        oof_path = os.path.join(root, 'oof_predictions_all.csv')
        oof_all.to_csv(oof_path, index=False)
        print(f'\n[OOF] 五折全量 Out-Of-Fold 预测已保存: {oof_path} (共 {len(oof_all)} 条样本)')

    print('\n' + '=' * 68)
    print('评估汇总 (Mean ± Std)')
    print('=' * 68)
    for _, r in summary.iterrows():
        print(f'  {r["metric"]:15s}: {r["mean"]} ± {r["std"]}')
    print('=' * 68)
    print(f'原始明细: {raw_path}')
    print(f'汇总表  : {summary_path}')
    print(f'文本报告: {report_path}')


if __name__ == '__main__':
    main()

