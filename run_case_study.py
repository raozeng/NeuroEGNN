# -*- coding: utf-8 -*-
"""
NeuroPred-3D-EGNN 实例分析

从五折测试预测结果中筛选典型案例:
  - 高置信正确 (TP/TN, 概率 > 0.9)
  - 高置信错误 (FP/FN, 概率偏离真实标签最远)
  - 边界案例 (概率在 0.4~0.6 之间)
输出为论文可用的表格和统计摘要。

产物 (默认 results/case_study/):
    case_study_cases.csv     案例明细表 (含序列/真实/预测/概率/来源折)
    case_study_summary.txt   统计摘要

用法:
    # 直接从五折测试预测结果中筛选 (无需重新推理)
    python run_case_study.py \
        --pred_dir results/5fold/results_5fold/test

    # 自定义阈值和每类数量
    python run_case_study.py \
        --pred_dir results/5fold/results_5fold/test \
        --high_conf 0.9 --n_per_class 10
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description='从五折测试预测中筛选典型案例')
    parser.add_argument('--pred_dir', type=str,
                        default=os.path.join('results', '5fold',
                                             'results_5fold', 'test'),
                        help='五折测试预测目录 (含 fold_<i>/test_predictions.csv)')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join('results', 'case_study'))
    parser.add_argument('--high_conf', type=float, default=0.9,
                        help='高置信阈值 (默认 0.9)')
    parser.add_argument('--low_conf_lo', type=float, default=0.4,
                        help='边界案例概率下限 (默认 0.4)')
    parser.add_argument('--low_conf_hi', type=float, default=0.6,
                        help='边界案例概率上限 (默认 0.6)')
    parser.add_argument('--n_per_class', type=int, default=10,
                        help='每类最多展示条数 (默认 10)')
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.join(SCRIPT_DIR, args.out_dir)
    os.makedirs(root, exist_ok=True)

    pred_dir = os.path.join(SCRIPT_DIR, args.pred_dir)

    # ---- 收集所有折的预测 ----
    all_dfs = []
    for i in range(10):
        csv_path = os.path.join(pred_dir, f'fold_{i}', 'test_predictions.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df['fold'] = i
        all_dfs.append(df)

    if not all_dfs:
        print(f'错误: 在 {pred_dir} 下未找到任何 fold_<i>/test_predictions.csv')
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)
    print(f'共加载 {len(df)} 条预测记录 (来自 {len(all_dfs)} 折)')

    # ---- 分类筛选 ----
    df['confidence'] = np.where(
        df['Predicted_Label'] == 1,
        df['Prediction_Probability'],
        1 - df['Prediction_Probability'])
    df['correct'] = df['True_Label'] == df['Predicted_Label']

    # 高置信正确
    tp = df[(df['True_Label'] == 1) & df['correct'] &
            (df['Prediction_Probability'] >= args.high_conf)]
    tn = df[(df['True_Label'] == 0) & df['correct'] &
            (df['Prediction_Probability'] <= 1 - args.high_conf)]
    # 高置信错误
    fp = df[(df['True_Label'] == 0) & ~df['correct'] &
            (df['Prediction_Probability'] >= args.high_conf)]
    fn = df[(df['True_Label'] == 1) & ~df['correct'] &
            (df['Prediction_Probability'] <= 1 - args.high_conf)]
    # 边界案例
    borderline = df[(df['Prediction_Probability'] >= args.low_conf_lo) &
                    (df['Prediction_Probability'] <= args.low_conf_hi)]

    n = args.n_per_class
    cases = []
    for label, sub, name in [
        (1, tp.nlargest(n, 'Prediction_Probability'), 'TP (高置信正确阳性)'),
        (0, tn.nsmallest(n, 'Prediction_Probability'), 'TN (高置信正确阴性)'),
        (0, fp.nlargest(n, 'Prediction_Probability'), 'FP (高置信误报)'),
        (1, fn.nsmallest(n, 'Prediction_Probability'), 'FN (高置信漏报)'),
        (-1, borderline.sample(min(n, len(borderline)), random_state=42),
         'Borderline (边界案例)'),
    ]:
        if len(sub) == 0:
            continue
        for _, row in sub.iterrows():
            cases.append({
                'Case_Type': name,
                'Sequence': row['Sequence'],
                'True_Label': int(row['True_Label']),
                'Predicted_Label': int(row['Predicted_Label']),
                'Probability': f'{row["Prediction_Probability"]:.4f}',
                'Confidence': f'{row["confidence"]:.4f}',
                'Fold': int(row['fold']),
            })

    cases_df = pd.DataFrame(cases)
    cases_path = os.path.join(root, 'case_study_cases.csv')
    cases_df.to_csv(cases_path, index=False, encoding='utf-8-sig')

    # ---- 统计摘要 ----
    lines = [
        'NeuroPred-3D-EGNN 实例分析摘要',
        '=' * 70,
        f'总预测数: {len(df)} | 来源: {args.pred_dir}',
        '',
        f'高置信阈值: {args.high_conf}',
        f'边界区间: [{args.low_conf_lo}, {args.low_conf_hi}]',
        '',
        '--- 各类型数量 ---',
        f'TP (高置信正确阳性, prob>={args.high_conf}): {len(tp)}',
        f'TN (高置信正确阴性, prob<={1-args.high_conf:.2f}): {len(tn)}',
        f'FP (高置信误报): {len(fp)}',
        f'FN (高置信漏报): {len(fn)}',
        f'Borderline (边界): {len(borderline)}',
        '',
        f'准确率: {df["correct"].mean():.4f}',
        f'平均置信度: {df["confidence"].mean():.4f}',
        '',
        '--- 置信度分布 ---',
        f'  >=0.9: {(df["confidence"] >= 0.9).sum()} '
        f'({(df["confidence"] >= 0.9).mean()*100:.1f}%)',
        f'  0.7-0.9: {((df["confidence"] >= 0.7) & (df["confidence"] < 0.9)).sum()} '
        f'({((df["confidence"] >= 0.7) & (df["confidence"] < 0.9)).mean()*100:.1f}%)',
        f'  0.5-0.7: {((df["confidence"] >= 0.5) & (df["confidence"] < 0.7)).sum()} '
        f'({((df["confidence"] >= 0.5) & (df["confidence"] < 0.7)).mean()*100:.1f}%)',
        f'  <0.5: {(df["confidence"] < 0.5).sum()} '
        f'({(df["confidence"] < 0.5).mean()*100:.1f}%)',
        '',
        f'案例明细: {cases_path}',
    ]

    # 添加各类型的代表性序列
    for case_type in cases_df['Case_Type'].unique():
        sub = cases_df[cases_df['Case_Type'] == case_type]
        lines.append(f'\n--- {case_type} (前 {min(5, len(sub))} 条) ---')
        for _, r in sub.head(5).iterrows():
            lines.append(f'  {r["Sequence"]}  true={r["True_Label"]} '
                         f'pred={r["Predicted_Label"]} '
                         f'prob={r["Probability"]} fold={r["Fold"]}')

    summary_path = os.path.join(root, 'case_study_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n' + '=' * 64)
    print('实例分析完成')
    print('=' * 64)
    print(f'案例明细: {cases_path} ({len(cases_df)} 条)')
    print(f'统计摘要: {summary_path}')
    print(f'  TP: {len(tp)} | TN: {len(tn)} | FP: {len(fp)} | '
          f'FN: {len(fn)} | Borderline: {len(borderline)}')


if __name__ == '__main__':
    main()
