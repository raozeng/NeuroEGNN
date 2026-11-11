# -*- coding: utf-8 -*-
"""
NeuroPred-3D-EGNN 一键流程入口

创新点一: 从 1D 序列迈向 3D 空间 —— ESMFold/结构先验 + E(n)-等变图神经网络
空间构象建模 + ESM 序列特征交叉注意力融合, 用于神经肽识别。

模式:
    prepare : 仅执行数据准备 (ESM-2 表征/接触图缓存 + 3D 结构重建缓存)
    train   : 训练 (自动触发数据准备)
    test    : 评估
    all     : 训练 + 评估 (默认)
    cv      : K 折交叉验证性能对比 (默认 5 折, 每折 1 份测试 + 其余 4 份按
              8:2 分训练/验证, 需先运行 make_split.py --folds 5 生成划分)

示例:
    python run.py                                   # 全流程 (默认小模型, CPU 可跑)
    python run.py --mode train --epochs 50          # 仅训练
    python run.py --mode test --model_path xxx.pth  # 仅评估
    python run.py --mode cv --epochs 50             # 五折交叉训练性能对比
    python run.py --use_api --api_limit 500         # 启用 ESMFold 在线 API 丰富结构
    python run.py --limit 300 --epochs 3            # 快速冒烟测试
"""

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
    parser = argparse.ArgumentParser(
        description='NeuroPred-3D-EGNN 全流程主入口 (3D 等变图 + ESM 交叉注意力融合)')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'train', 'test', 'prepare', 'cv'],
                        help='执行模式: all / train / test / prepare / cv')
    parser.add_argument('--folds', type=int, default=5,
                        help='cv 模式: 交叉验证折数 (默认 5)')
    parser.add_argument('--cv_dir', type=str,
                        default=os.path.join('results', '5fold',
                                             'results_5fold'),
                        help='cv 模式: 结果根目录 (默认 '
                             'results/5fold/results_5fold, 其下按 '
                             'train/fold_0..、test/fold_0.. 分目录保存)')
    parser.add_argument('--train_csv', type=str, default='training_v2.csv')
    parser.add_argument('--val_csv', type=str, default='validation_v2.csv',
                        help='独立验证集 (早停/选模), 由 make_split.py 生成')
    parser.add_argument('--test_csv', type=str, default='testing_v2.csv',
                        help='最终 held-out 测试集, 训练全程不接触')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--esm_model', type=str, default='esm2_t33_650M_UR50D',
                        help='ESM-2 型号: esm2_t33_650M_UR50D (默认论文级配置, 已缓存) / '
                             'esm2_t6_8M_UR50D (轻量冒烟测试)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型权重路径 (默认: 脚本目录下 neuro3d_best_model.pth)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='预测明细输出 CSV')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=500)
    parser.add_argument('--use_api', action='store_true',
                        help='使用 ESMFold 在线 API 预测真实 3D 结构 (需联网)')
    parser.add_argument('--api_limit', type=int, default=0,
                        help='ESMFold API 调用条数上限 (0=不限制)')
    parser.add_argument('--limit', type=int, default=0,
                        help='调试: 训练/测试仅取前 N 条 (0=全量)')
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    print('\n' + '=' * 64)
    print('        NeuroPred-3D-EGNN  |  创新点一: 1D 序列 -> 3D 空间')
    print('   ESMFold 结构先验 + E(n)-等变图神经网络 + 交叉注意力融合')
    print('=' * 64)
    print(f"  模式: {args.mode.upper()} | 设备: {args.device or '自动检测 (优先GPU)'}")
    print(f"  ESM: {args.esm_model} | ESMFold API: {'开启' if args.use_api else '关闭 (离线回退: 接触图MDS)'}")
    print('=' * 64 + '\n')

    # 0. 仅数据准备
    if args.mode == 'prepare':
        for csv_path in (args.train_csv, args.test_csv):
            print(f">>> 数据准备: {resolve_data_path(csv_path)}")
            prepare_esm_cache(resolve_data_path(csv_path), os.path.join(SCRIPT_DIR, 'cache'),
                              esm_model=args.esm_model, max_len=50,
                              use_api=args.use_api, api_limit=args.api_limit)
        print('>>> 数据准备完成 (缓存位于 NeuroPred-3D-EGNN/cache/)')
        return

    # 0.5 K 折交叉验证: 每折 1 份测试 + 其余 4 份按 8:2 分训练/验证
    #     产物目录 (与老师模板 results/5fold/results_5fold_davis/... 一致):
    #       results/5fold/results_5fold/train/fold_0/best_model.pth
    #                                            /latest_checkpoint.pth
    #                                            /training_log.csv
    #                                            /val_metrics.txt
    #                                            /val_predictions.csv
    #       results/5fold/results_5fold/test/fold_0/metrics.txt
    #                                           /test_predictions.csv
    #       ... fold_1 .. fold_4 ...
    #       注: 目录 fold_0 对应数据文件 training/validation/testing_v2_fold1.csv,
    #           fold_1 对应 fold2.csv, 依此类推 (0 起始编号与老师模板对齐)。
    #       results/5fold/results_5fold/cv_results.csv  (逐折指标 + Mean±Std)
    #       results/5fold/results_5fold/mean_std.txt    (最终平均值 ± 标准差)
    if args.mode == 'cv':
        import pandas as pd
        from train import format_metrics_text

        results_root = os.path.join(SCRIPT_DIR, args.cv_dir)
        train_root = os.path.join(results_root, 'train')
        test_root = os.path.join(results_root, 'test')
        os.makedirs(train_root, exist_ok=True)
        os.makedirs(test_root, exist_ok=True)

        rows = []
        # 目录折编号与老师模板一致: fold_0 .. fold_{K-1} (0 起始);
        # 数据文件仍为 make_split.py 生成的 fold1 .. foldK (1 起始),
        # 对应关系: fold_0 -> *_fold1.csv, fold_1 -> *_fold2.csv, ...
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
            test_pred_csv = os.path.join(fold_test_dir,
                                         'test_predictions.csv')
            test_metrics_txt = os.path.join(fold_test_dir, 'metrics.txt')

            print(f'\n>>> [CV] 折 {fold_idx}/{args.folds - 1} '
                  f'(数据文件 fold{fold}) | '
                  f'train: {train_csv} | val: {val_csv} | test: {test_csv}')
            print(f'>>> [CV] 训练产物目录: {fold_train_dir}')
            print(f'>>> [CV] 测试产物目录: {fold_test_dir}')

            # ---- 本折训练 (内部保存 best/latest/log/val 五件套) ----
            train(train_csv=train_csv, val_csv=val_csv, epochs=args.epochs,
                  batch_size=args.batch_size, lr=args.lr, save_path=best_model,
                  device=args.device, seed=args.seed, esm_model=args.esm_model,
                  use_api=args.use_api, api_limit=args.api_limit,
                  limit=args.limit, out_dir=fold_train_dir)

            # ---- 本折测试 (逐条预测 + 指标文本) ----
            metrics, _ = test(test_csv=test_csv, model_path=best_model,
                              batch_size=args.batch_size,
                              output_csv=test_pred_csv, device=args.device,
                              limit=args.limit)
            with open(test_metrics_txt, 'w', encoding='utf-8') as f:
                f.write(f"Fold {fold_idx} test metrics "
                        f"(数据文件: {test_csv})\n"
                        f"train: {train_csv} | val: {val_csv} | "
                        f"test: {test_csv}\n")
                f.write(format_metrics_text(
                    metrics, f'Test Metrics (fold_{fold_idx})'))
            print(f">>> [CV] 本折测试指标已保存: {test_metrics_txt}")

            rows.append({'Fold': fold_idx, 'train_csv': train_csv,
                         'val_csv': val_csv, 'test_csv': test_csv, **metrics})

        # ---- 汇总: 逐折指标 + Mean ± Std ----
        df = pd.DataFrame(rows)
        metric_cols = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity',
                       'MCC', 'AUC', 'F1-Score']
        count_cols = ['TP', 'TN', 'FP', 'FN']
        float_cols = [c for c in df.columns if df[c].dtype.kind == 'f']

        stat_row = {'Fold': 'Mean±Std',
                    'train_csv': '', 'val_csv': '', 'test_csv': ''}
        for c in float_cols:
            stat_row[c] = (f"{df[c].mean():.4f}±{df[c].std(ddof=1):.4f}")
        for c in count_cols:
            if c in df.columns:
                stat_row[c] = int(df[c].sum())
        df_out = pd.concat([df, pd.DataFrame([stat_row])],
                           ignore_index=True)
        results_path = os.path.join(results_root, 'cv_results.csv')
        df_out.to_csv(results_path, index=False)

        # 纯文本版最终结果 (方便直接贴进报告)
        mean_std_path = os.path.join(results_root, 'mean_std.txt')
        with open(mean_std_path, 'w', encoding='utf-8') as f:
            f.write(f'{args.folds}-fold cross-validation (Mean ± Std)\n')
            f.write('=' * 50 + '\n')
            for c in metric_cols:
                f.write(f"{c:12s}: {df[c].mean():.4f} ± "
                        f"{df[c].std(ddof=1):.4f}\n")
            f.write('-' * 50 + '\n')
            f.write('Per-fold metrics:\n')
            f.write(df[['Fold'] + metric_cols].to_string(index=False))
            f.write('\n')

        print('\n' + '=' * 64)
        print(f'        五折交叉验证性能对比 ({args.folds} 折, 每折独立测试集)')
        print('=' * 64)
        print(df[['Fold'] + metric_cols + count_cols].to_string(index=False))
        print('-' * 64)
        print(' Mean ± Std:')
        for c in metric_cols:
            print(f"   {c:12s}: {df[c].mean():.4f} ± {df[c].std(ddof=1):.4f}")
        print('=' * 64)
        print(f" 逐折汇总已保存: {results_path}")
        print(f" 均值±标准差已保存: {mean_std_path}\n")
        return

    # 1. 训练
    if args.mode in ('all', 'train'):
        print('>>> 步骤 1/2: 启动 3D-EGNN 模型训练与验证')
        result = train(train_csv=args.train_csv, val_csv=args.val_csv,
                       epochs=args.epochs, batch_size=args.batch_size,
                       lr=args.lr, save_path=args.model_path,
                       device=args.device, seed=args.seed,
                       esm_model=args.esm_model, use_api=args.use_api,
                       api_limit=args.api_limit, limit=args.limit)
        print(f">>> 训练完成, 最佳验证准确率: {result['best_acc']:.4f}\n")

    # 2. 评估
    if args.mode in ('all', 'test'):
        print('>>> 步骤 2/2: 启动测试集评估与预测导出')
        metrics, results_df = test(test_csv=args.test_csv,
                                   model_path=args.model_path,
                                   batch_size=args.batch_size,
                                   output_csv=args.output_csv,
                                   device=args.device,
                                   limit=args.limit)
        print(f">>> 评估完成, 测试集 Accuracy: {metrics['Accuracy']:.4f}, "
              f"AUC: {metrics['AUC']:.4f}\n")

    elapsed = time.time() - start_time
    print('=' * 64)
    print(f' 全流程执行结束! 总耗时: {elapsed:.1f} 秒')
    print('=' * 64 + '\n')


if __name__ == '__main__':
    main()
