# -*- coding: utf-8 -*-
"""
NeuroPred-3D-EGNN 消融实验运行器

消融变体 (见 model.ABLATION_PRESETS):
    full     : 完整模型 (ESM-2 + 3D-EGNN + 双向交叉注意力)
    no_3d    : 去掉 3D-EGNN 空间分支 (仅 ESM 序列)
    no_esm   : 去掉 ESM 序列语义 (仅 one-hot/理化 + 3D 图)
    no_attn  : 双向交叉注意力 -> 两模态等权平均
    no_coord : EGNN 去掉 E(n) 等变坐标更新 (静态坐标)

评估协议:
    固定单划分 training_v2.csv / validation_v2.csv / testing_v2.csv,
    每个变体用相同的若干随机种子各训练一次, 报告 mean ± std;
    验证集仅用于早停/选模, 测试集全程不参与训练。

产物 (默认 results/ablation/):
    <variant>/seed_<s>/train/best_model.pth, latest_checkpoint.pth,
                              training_log.csv, val_metrics.txt, val_predictions.csv
    <variant>/seed_<s>/test/metrics.txt, test_predictions.csv
    ablation_raw.csv      逐次运行原始指标 (每个 变体×种子 一行)
    ablation_summary.csv  每变体 mean ± std 汇总
    ablation_report.txt   论文用对比表 (含相对 full 的 Δ)

示例:
    # 正式实验 (服务器 GPU, ESM t33, 与五折口径一致, 5 变体 × 3 种子)
    python run_ablation.py --esm_model esm2_t33_650M_UR50D --epochs 50

    # 只跑某几个变体 / 指定种子
    python run_ablation.py --modes full no_3d --seeds 500 501

    # 冒烟: 小模型 + 少量数据 + 2 轮, 验证全链路
    python run_ablation.py --esm_model esm2_t6_8M_UR50D --limit 200 --epochs 2

    # 重跑已完成的单元 (默认断点续跑: 已有 best_model + 测试产物则跳过)
    python run_ablation.py --force
"""

import os
import sys
import time
import argparse
import traceback

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train import train, format_metrics_text
from test import test
from model import ABLATION_PRESETS

METRIC_COLS = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity',
               'MCC', 'AUC', 'F1-Score']
COUNT_COLS = ['TP', 'TN', 'FP', 'FN']


def parse_args():
    parser = argparse.ArgumentParser(
        description='NeuroPred-3D-EGNN 消融实验运行器 (5 变体 × N 种子)')
    parser.add_argument('--modes', nargs='+', default=list(ABLATION_PRESETS),
                        choices=list(ABLATION_PRESETS),
                        help='要跑的消融变体 (默认全部 5 个)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[500, 501, 502],
                        help='随机种子列表 (默认 500 501 502, 每变体各一次)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--esm_model', type=str, default='esm2_t33_650M_UR50D',
                        help='ESM-2 型号 (正式实验建议 t33_650M, 与五折一致; '
                             '冒烟可用 t6_8M)')
    parser.add_argument('--train_csv', type=str, default='training_v2.csv')
    parser.add_argument('--val_csv', type=str, default='validation_v2.csv')
    parser.add_argument('--test_csv', type=str, default='testing_v2.csv')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join('results', 'ablation'),
                        help='消融结果根目录 (默认 results/ablation)')
    parser.add_argument('--limit', type=int, default=0,
                        help='调试: 每集仅取前 N 条 (0=全量)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--force', action='store_true',
                        help='强制重跑: 忽略已存在的模型/测试产物')
    return parser.parse_args()


def run_one(variant, seed, args, root):
    """单个 (变体, 种子) 单元: 训练 -> 测试 -> 指标行。已完成则跳过。"""
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
        print(f'\n>>> [跳过] {variant} / seed {seed}: 产物已存在 '
              f'(加 --force 可重跑)')
        prev = pd.read_csv(test_pred_csv)
        # 从已保存的 metrics.txt 不易结构化解析, 直接用预测 CSV 重算
        from train import full_metrics
        metrics = full_metrics(prev['True_Label'].values,
                               prev['Predicted_Label'].values,
                               prev['Prediction_Probability'].values)
        ckpt = __import__('torch').load(best_model, map_location='cpu',
                                        weights_only=False)
        n_params = int(sum(v.numel() for v in ckpt['state_dict'].values()))
        return {'variant': variant, 'seed': seed, 'n_params': n_params,
                'status': 'cached', **metrics}

    print('\n' + '#' * 64)
    print(f'# 消融单元: {variant} | seed {seed}')
    print(f'# 开关: {ABLATION_PRESETS[variant]}')
    print('#' * 64)
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
        f.write(f'Ablation: {variant} | seed: {seed} | '
                f'ESM: {args.esm_model}\n')
        f.write(f'train: {args.train_csv} | val: {args.val_csv} | '
                f'test: {args.test_csv}\n')
        f.write(format_metrics_text(metrics, f'Test Metrics ({variant})'))

    print(f'>>> 单元完成 {variant}/seed{seed}, '
        f'耗时 {time.time() - t0:.0f}s, AUC={metrics["AUC"]:.4f}')
    return {'variant': variant, 'seed': seed,
            'n_params': int(train_result['n_params']),
            'status': 'trained', **metrics}


def write_outputs(rows, args, root):
    """raw 明细 + mean±std 汇总 + 论文格式文本报告。"""
    raw_path = os.path.join(root, 'ablation_raw.csv')
    summary_path = os.path.join(root, 'ablation_summary.csv')
    report_path = os.path.join(root, 'ablation_report.txt')

    raw = pd.DataFrame(rows)
    # 固定列顺序
    raw = raw[['variant', 'seed', 'n_params', 'status'] + METRIC_COLS +
              COUNT_COLS]
    raw.to_csv(raw_path, index=False, encoding='utf-8-sig')

    # ---- mean ± std ----
    summary_rows = []
    for variant in args.modes:
        sub = raw[raw['variant'] == variant]
        row = {'variant': variant, 'n_runs': len(sub),
               'n_params': int(sub['n_params'].iloc[0]) if len(sub) else 0}
        for c in METRIC_COLS:
            vals = sub[c].astype(float)
            row[f'{c}_mean'] = vals.mean()
            row[f'{c}_std'] = vals.std(ddof=1) if len(vals) > 1 else 0.0
            row[c] = f'{vals.mean():.4f}±{vals.std(ddof=1):.4f}' \
                if len(vals) > 1 else f'{vals.mean():.4f}'
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')

    # ---- 文本报告 (含相对 full 的 Δ) ----
    lines = []
    lines.append('NeuroPred-3D-EGNN 消融实验报告')
    lines.append('=' * 90)
    lines.append(f'划分: train={args.train_csv} | val={args.val_csv} | '
                 f'test={args.test_csv}')
    lines.append(f'ESM: {args.esm_model} | epochs: {args.epochs} | '
                 f'batch_size: {args.batch_size} | lr: {args.lr} | '
                 f'seeds: {args.seeds}')
    lines.append('')
    header = (f'{"Variant":10s} {"Params":>8s}  ' +
              '  '.join(f'{c:>11s}' for c in METRIC_COLS))
    lines.append(header)
    lines.append('-' * len(header))
    for _, r in summary.iterrows():
        lines.append(
            f'{r["variant"]:10s} {r["n_params"]/1e6:7.3f}M  ' +
            '  '.join(f'{r[c]:>11s}' for c in METRIC_COLS))
    lines.append('')

    # Δ vs full (按同种子配对求差, 再平均; 正值=消融后反而升高)
    full = raw[raw['variant'] == 'full']
    if len(full):
        lines.append('相对完整模型 (full) 的指标变化 Δ = variant - full '
                     '(同种子配对, 负值表示该组件有正贡献)')
        lines.append('-' * 90)
        lines.append(f'{"Variant":10s} ' +
                     '  '.join(f'{("Δ"+c):>11s}' for c in METRIC_COLS))
        for variant in args.modes:
            if variant == 'full':
                continue
            sub = raw[raw['variant'] == variant]
            merged = sub.merge(full, on='seed', suffixes=('', '_full'))
            if not len(merged):
                continue
            deltas = []
            for c in METRIC_COLS:
                d = (merged[c] - merged[f'{c}_full']).mean()
                deltas.append(f'{d:+.4f}')
            lines.append(f'{variant:10s} ' +
                         '  '.join(f'{d:>11s}' for d in deltas))
        lines.append('')

    lines.append('逐次运行原始结果:')
    lines.append('-' * 90)
    show = raw[['variant', 'seed'] + METRIC_COLS].copy()
    lines.append(show.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    # ---- 控制台同步打印核心表 ----
    print('\n' + '=' * 90)
    print('消融实验完成 (Mean ± Std)')
    print('=' * 90)
    print(header)
    print('-' * len(header))
    for _, r in summary.iterrows():
        print(f'{r["variant"]:10s} {r["n_params"]/1e6:7.3f}M  ' +
              '  '.join(f'{r[c]:>11s}' for c in METRIC_COLS))
    print('=' * 90)
    print(f'原始明细: {raw_path}')
    print(f'汇总表  : {summary_path}')
    print(f'文本报告: {report_path}')


def main():
    args = parse_args()
    root = os.path.join(SCRIPT_DIR, args.out_dir)
    os.makedirs(root, exist_ok=True)

    print('=' * 64)
    print('  NeuroPred-3D-EGNN 消融实验')
    print(f'  变体: {args.modes}')
    print(f'  种子: {args.seeds}  (共 {len(args.modes) * len(args.seeds)} '
          f'个训练单元)')
    print(f'  ESM: {args.esm_model} | epochs: {args.epochs} | '
          f'设备: {args.device or "自动"}')
    print(f'  结果目录: {root}')
    print('=' * 64)

    rows = []
    t_start = time.time()
    for variant in args.modes:
        for seed in args.seeds:
            try:
                row = run_one(variant, seed, args, root)
                rows.append(row)
            except Exception as e:
                # 单单元失败不影响其余单元; 错误落盘后继续
                traceback.print_exc()
                rows.append({'variant': variant, 'seed': seed,
                             'n_params': 0, 'status': f'failed: {e}',
                             **{c: np.nan for c in METRIC_COLS},
                             **{c: np.nan for c in COUNT_COLS}})
            # 增量落盘, 长任务中断也不丢已完成结果
            if rows:
                pd.DataFrame(rows).to_csv(
                    os.path.join(root, 'ablation_raw.csv'),
                    index=False, encoding='utf-8-sig')

    ok = [r for r in rows if not str(r['status']).startswith('failed')]
    if ok:
        write_outputs(rows, args, root)
    print(f'\n全部单元结束: 成功 {len(ok)}/{len(rows)}, '
          f'总耗时 {(time.time() - t_start) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
