# -*- coding: utf-8 -*-
"""
NeuroPred-3D-EGNN 训练脚本

训练协议与 baseline 保持一致 (Softmax 输出 + one-hot BCE 损失), 指标对齐,
便于公平对比 1D-TCN 方案与本方案 (3D-EGNN + 交叉注意力融合)。

用法:
    python train.py --epochs 50 --batch_size 128
    python train.py --limit 300 --epochs 3          # 快速冒烟
"""

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
    """Accuracy / Sensitivity / Specificity / AUC / MCC / F1 (与 baseline 对齐)。"""
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
    """与 test.py 对齐的全套指标 (含 Precision 与混淆矩阵计数)。"""
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
    """把指标字典渲染成与 test.py 控制台报表一致的纯文本。"""
    lines = ['-' * 49, title, '-' * 49,
             f"Accuracy                      : {metrics['Accuracy']:.4f}",
             f"Precision                     : {metrics['Precision']:.4f}",
             f"Sensitivity (Recall)          : {metrics['Sensitivity']:.4f}",
             f"Specificity                   : {metrics['Specificity']:.4f}",
             f"Matthews Corr Coef (MCC)      : {metrics['MCC']:.4f}",
             f"AUC                           : {metrics['AUC']:.4f}",
             f"F1-Score                      : {metrics['F1-Score']:.4f}",
             f"Confusion Matrix (TP/TN/FP/FN): "
             f"{metrics['TP']} / {metrics['TN']} / {metrics['FP']} / {metrics['FN']}",
             '-' * 49]
    return '\n'.join(lines) + '\n'


@torch.no_grad()
def predict(model, dataloader, device):
    """对给定数据集做纯推理, 返回序列/真实标签/预测标签/正类概率。"""
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
    """单轮训练或验证。label_smoothing>0 时对 one-hot 目标做平滑, 抑制过度自信。"""
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
                target = (target * (1.0 - label_smoothing) +
                          label_smoothing / 2.0)
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
    """训练主函数: 确保缓存 -> 构建数据加载器 -> 逐轮训练 + 独立验证集做模型选择。

    ablation: 消融变体名 (full/no_3d/no_esm/no_attn/no_coord),
        见 model.ABLATION_PRESETS; 开关会写入 checkpoint config,
        test.py 评估时自动按相同变体构建模型。

    out_dir 不为 None 时 (五折交叉验证每折调用), 在该目录下保存:
        best_model.pth         验证集 AUC 最优的模型权重 (用于测试集评估)
        latest_checkpoint.pth  最后一轮 checkpoint (含优化器/调度器状态, 可断点续训)
        training_log.csv       逐 epoch 训练/验证指标日志
        val_metrics.txt        最优模型在验证集上的全套指标
        val_predictions.csv    验证集每个样本的真实标签/预测标签/正类概率
    单折旧用法 (out_dir=None) 的保存行为保持不变。

    抗过拟合协议:
        - train / validation / testing 三集同源不重叠 (见 make_split.py),
          验证集仅用于早停与最优模型选择, 不接触测试集;
        - 以验证集 AUC (而非 Acc) 严格提升作为保存/早停判据, 减少阈值噪声;
        - label smoothing + 较高 dropout/weight_decay + 较小学习率。
    """
    set_seed(seed)
    if ablation not in ABLATION_PRESETS:
        raise ValueError(
            f"未知消融变体 '{ablation}', 可选: {list(ABLATION_PRESETS)}")
    ablation_flags = ABLATION_PRESETS[ablation]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    # ---- 每折产物目录 (五折交叉验证) ----
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

    print('=' * 60)
    print(f" NeuroPred-3D-EGNN 训练 | 设备: {device} | 批大小: {batch_size} | lr: {lr}")
    print(f" 消融变体: {ablation} | 开关: {ablation_flags}")
    print(f" ESM: {esm_model} | EGNN: {egnn_layers} 层 x {hidden_dim} 维 | kNN: {knn} | 距离阈值: {contact_dist}Å")
    print(f" 训练集: {train_csv} | 验证集: {val_csv}")
    print(f" 最优权重: {save_path}")
    print('=' * 60)

    # ---- 数据缓存与加载器 ----
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
    print(f" 训练 batch 数: {len(train_loader)} | 验证 batch 数: {len(val_loader)}")

    # ---- 模型 (按消融预设构建; 关闭的模块整组不存在) ----
    model = NeuroPred3D(esm_dim=_read_esm_dim(train_npz), hidden_dim=hidden_dim,
                        egnn_layers=egnn_layers, heads=heads, dropout=dropout,
                        **ablation_flags)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" 可训练参数量: {n_params / 1e6:.2f} M")

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1))

    # ---- 训练循环 (验证集 AUC 严格提升才保存, 并在内存保留最优权重) ----
    best_auc, best_acc, best_metrics, bad_epochs = 0.0, 0.0, {}, 0
    best_state = None
    history = []
    for epoch in range(epochs):
        train_m = run_epoch(model, train_loader, criterion, optimizer,
                            device, train_mode=True,
                            label_smoothing=label_smoothing)
        val_m = run_epoch(model, val_loader, criterion, None,
                          device, train_mode=False)
        scheduler.step()

        saved_tag = ''
        if val_m['auc'] > best_auc + 1e-6:
            best_auc = val_m['auc']
            best_acc = val_m['acc']
            best_metrics = val_m.copy()
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            torch.save({'state_dict': best_state,
                        'config': {**model.config, 'ablation': ablation,
                                   'esm_model': esm_model,
                                   'max_len': max_len, 'knn': knn,
                                   'contact_dist': contact_dist}},
                       save_path)
            saved_tag = ' -> [已保存最优模型]'
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

        # ---- 逐 epoch 日志 + 最新 checkpoint (可断点续训) ----
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
                'state_dict': {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auc': best_auc,
                'config': {**model.config, 'ablation': ablation,
                           'esm_model': esm_model,
                           'max_len': max_len, 'knn': knn,
                           'contact_dist': contact_dist},
            }, latest_path)

        if bad_epochs >= patience:
            print(f"[早停] 验证集 AUC 连续 {patience} 轮未提升, 提前结束训练。")
            break

    # 恢复验证集最优权重, 保证最终模型即最优模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- 最优模型在验证集上的全套指标 + 逐条预测 (五折每折必存) ----
    if val_metrics_path is not None and val_pred_path is not None:
        val_seqs, val_labels, val_preds, val_probs = predict(
            model, val_loader, device)
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
        print(f"[保存] 验证集指标: {val_metrics_path}")
        print(f"[保存] 验证集逐条预测: {val_pred_path} "
              f"(共 {len(val_labels)} 条)")
        print(f"[保存] 训练日志: {log_path} | 最新 checkpoint: {latest_path}")

    print('=' * 60)
    print(f"训练完成! 最佳验证 AUC: {best_auc:.4f} (对应 Acc: {best_acc:.4f})")
    print(f"最优模型: {save_path}")
    print('=' * 60)
    return {'best_auc': best_auc, 'best_acc': best_acc,
            'best_metrics': best_metrics, 'save_path': save_path,
            'n_params': n_params}


def _read_esm_dim(npz_path):
    import numpy as np
    with np.load(npz_path, allow_pickle=True) as data:
        return int(data['esm_dim'])


def parse_args():
    parser = argparse.ArgumentParser(description='NeuroPred-3D-EGNN 训练脚本')
    parser.add_argument('--train_csv', type=str, default='training_v2.csv')
    parser.add_argument('--val_csv', type=str, default='validation_v2.csv',
                        help='独立验证集 (仅用于早停/选模, 勿与测试集混用)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--esm_model', type=str, default='esm2_t33_650M_UR50D',
                        help="ESM-2 型号 (默认论文配置: esm2_t33_650M_UR50D; "
                             "轻量冒烟可选 esm2_t6_8M_UR50D)")
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--knn', type=int, default=8, help='kNN 空间近邻数')
    parser.add_argument('--contact_dist', type=float, default=10.0,
                        help='残基接触距离阈值 (Å)')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--egnn_layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=15, help='早停耐心轮数')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                        help='one-hot BCE 标签平滑系数 (0=关闭)')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=500)
    parser.add_argument('--use_api', action='store_true',
                        help='尝试调用 ESMFold 在线 API 获取真实结构 (需联网)')
    parser.add_argument('--api_limit', type=int, default=0,
                        help='API 调用上限 (0=不限制)')
    parser.add_argument('--limit', type=int, default=0,
                        help='调试用: 仅使用前 N 条训练数据 (0=全量)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='五折交叉验证: 本折训练产物目录 '
                             '(best_model/latest_checkpoint/training_log/'
                             'val_metrics/val_predictions)')
    parser.add_argument('--ablation', type=str, default='full',
                        choices=list(ABLATION_PRESETS),
                        help='消融变体: full(完整) / no_3d(去3D-EGNN) / '
                             'no_esm(去ESM) / no_attn(去交叉注意力) / '
                             'no_coord(去等变坐标更新)')
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
          use_api=args.use_api,
          api_limit=args.api_limit, limit=args.limit, out_dir=args.out_dir,
          ablation=args.ablation)
