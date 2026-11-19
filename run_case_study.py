# -*- coding: utf-8 -*-
"""Select representative high-confidence, error, and borderline predictions from K-fold outputs."""

import os
import sys
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='Extract representative case-study peptides from K-fold predictions')
    parser.add_argument('--pred_dir', type=str,
                        default=os.path.join('results', '5fold', 'results_5fold', 'test'),
                        help='Directory containing fold_<i>/test_predictions.csv')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join('results', 'case_study'))
    parser.add_argument('--high_conf', type=float, default=0.9)
    parser.add_argument('--low_conf_lo', type=float, default=0.4)
    parser.add_argument('--low_conf_hi', type=float, default=0.6)
    parser.add_argument('--n_per_class', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.join(SCRIPT_DIR, args.out_dir)
    os.makedirs(root, exist_ok=True)
    pred_dir = os.path.join(SCRIPT_DIR, args.pred_dir)

    all_dfs = []
    for i in range(10):
        csv_path = os.path.join(pred_dir, f'fold_{i}', 'test_predictions.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df['fold'] = i
        all_dfs.append(df)

    if not all_dfs:
        print(f"Error: no fold_<i>/test_predictions.csv found under {pred_dir}")
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)
    df['confidence'] = np.where(
        df['Predicted_Label'] == 1,
        df['Prediction_Probability'],
        1 - df['Prediction_Probability'])
    df['correct'] = df['True_Label'] == df['Predicted_Label']

    tp = df[(df['True_Label'] == 1) & df['correct'] &
            (df['Prediction_Probability'] >= args.high_conf)]
    tn = df[(df['True_Label'] == 0) & df['correct'] &
            (df['Prediction_Probability'] <= 1 - args.high_conf)]
    fp = df[(df['True_Label'] == 0) & ~df['correct'] &
            (df['Prediction_Probability'] >= args.high_conf)]
    fn = df[(df['True_Label'] == 1) & ~df['correct'] &
            (df['Prediction_Probability'] <= 1 - args.high_conf)]
    borderline = df[(df['Prediction_Probability'] >= args.low_conf_lo) &
                    (df['Prediction_Probability'] <= args.low_conf_hi)]

    n = args.n_per_class
    cases = []
    for sub, name in [
        (tp.nlargest(n, 'Prediction_Probability'), 'High-Confidence TP'),
        (tn.nsmallest(n, 'Prediction_Probability'), 'High-Confidence TN'),
        (fp.nlargest(n, 'Prediction_Probability'), 'High-Confidence FP'),
        (fn.nsmallest(n, 'Prediction_Probability'), 'High-Confidence FN'),
        (borderline.sample(min(n, len(borderline)), random_state=42), 'Borderline'),
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

    lines = [
        'NeuroEGNN Case Study Summary',
        '=' * 70,
        f'Total predictions: {len(df)} | Source: {args.pred_dir}',
        f'High-confidence threshold: {args.high_conf} | Borderline interval: [{args.low_conf_lo}, {args.low_conf_hi}]',
        '',
        f'High-Confidence TP (P >= {args.high_conf}): {len(tp)}',
        f'High-Confidence TN (P <= {1 - args.high_conf:.2f}): {len(tn)}',
        f'High-Confidence FP: {len(fp)}',
        f'High-Confidence FN: {len(fn)}',
        f'Borderline: {len(borderline)}',
        f'Overall accuracy: {df["correct"].mean():.4f} | Mean confidence: {df["confidence"].mean():.4f}',
    ]
    for case_type in cases_df['Case_Type'].unique():
        sub = cases_df[cases_df['Case_Type'] == case_type]
        lines.append(f'\n[{case_type}]')
        for _, r in sub.head(5).iterrows():
            lines.append(f'  {r["Sequence"]}  true={r["True_Label"]} '
                         f'pred={r["Predicted_Label"]} prob={r["Probability"]} fold={r["Fold"]}')

    summary_path = os.path.join(root, 'case_study_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"[CaseStudy] Saved {cases_path} ({len(cases_df)} cases) and {summary_path}")


if __name__ == '__main__':
    main()
