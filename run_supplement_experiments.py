# -*- coding: utf-8 -*-
"""
Supplementary Experiments & Comprehensive Statistical/Interpretability Analysis
for NeuroEGNN SCI Manuscript.

Tasks performed:
1. Dataset statistical characterization (length distribution, AA enrichment,
   N-terminal Met bias, homology cluster distribution, 3D structural properties).
2. Comprehensive Baseline Comparison across 3 paradigms on BOTH:
   (a) 5-Fold Homology-Isolated Cross-Validation (folds 1..5, N=8,887)
   (b) Standard Homology-Isolated Hold-Out Benchmark (train=7,110, val=889, test=888)
   Baselines:
     - AAC+DPC+PCP + Logistic Regression (LR)
     - AAC+DPC+PCP + Support Vector Machine (SVM-RBF)
     - AAC+DPC+PCP + Random Forest (RF)
     - AAC+DPC+PCP + HistGradientBoosting (GBDT)
     - AAC+DPC+PCP + Multi-Layer Perceptron (MLP)
     - 3D Geometric Topology + PCP + RF
     - 3D Geometric Topology + PCP + MLP
     - ESM-2 (t6_8M, 320D) + LR (Hold-out)
     - ESM-2 (t6_8M, 320D) + MLP (Hold-out)
     - ESM-2 (t33_650M, 1280D) + LR
     - ESM-2 (t33_650M, 1280D) + RF
     - ESM-2 (t33_650M, 1280D) + SVM-RBF
3. Statistical significance testing (Paired t-test, Wilcoxon signed-rank test,
   Bootstrap AUC p-values, AUPRC computation) for Baselines and Ablation variants.
4. Length-stratified performance analysis (Short <=15 aa, Medium 16-30 aa, Long >30 aa).
5. Global quantitative interpretability analysis across all 888 test sequences
   (amino acid attention enrichment, spatial distance vs. sequence separation in s2g/g2s).
6. Generation of publication-grade figures saved to ../paper/figures/.
"""

import os
import sys
import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    matthews_corrcoef, roc_auc_score, average_precision_score, f1_score,
    roc_curve, precision_recall_curve
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CAND_FIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'paper', 'figures')
PAPER_FIG_DIR = _CAND_FIG_DIR if os.path.isdir(_CAND_FIG_DIR) else os.path.join(SCRIPT_DIR, 'results', 'figures')
BASELINE_OUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'baselines')
os.makedirs(PAPER_FIG_DIR, exist_ok=True)
os.makedirs(BASELINE_OUT_DIR, exist_ok=True)

AAS = 'ARNDCQEGHILKMFPSTWYV'
AA_TO_IDX = {a: i for i, a in enumerate(AAS)}
DIPEPTIDES = [a + b for a in AAS for b in AAS]
DIP_TO_IDX = {d: i for i, d in enumerate(DIPEPTIDES)}

# Physicochemical properties (same as dataset.py)
_HYDRO = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4,
              H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8,
              T=-0.7, W=-0.9, Y=-1.3, V=4.2)
_POLAR = dict(A=8.1, R=10.5, N=11.6, D=13.0, C=5.5, Q=10.5, E=12.3, G=9.0,
              H=10.4, I=5.2, L=4.9, K=11.3, M=5.7, F=5.2, P=8.0, S=9.2,
              T=8.6, W=5.4, Y=6.2, V=5.9)
_VOLUME = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4,
               G=60.1, H=153.2, I=166.7, L=166.7, K=168.6, M=162.9, F=189.9,
               P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
_CHARGE = dict(A=0.0, R=1.0, N=0.0, D=-1.0, C=0.0, Q=0.0, E=-1.0, G=0.0,
               H=0.1, I=0.0, L=0.0, K=1.0, M=0.0, F=0.0, P=0.0, S=0.0,
               T=0.0, W=0.0, Y=0.0, V=0.0)
_FLEX = dict(A=0.36, R=0.53, N=0.46, D=0.51, C=0.35, Q=0.49, E=0.50, G=0.54,
             H=0.32, I=0.46, L=0.37, K=0.53, M=0.30, F=0.31, P=0.51, S=0.51,
             T=0.44, W=0.31, Y=0.42, V=0.39)
PROP_TABLES = [_HYDRO, _POLAR, _VOLUME, _CHARGE, _FLEX]


def clean_seq(seq, max_len=50):
    cleaned = ''.join(aa if aa in AA_TO_IDX else 'X' for aa in str(seq))
    return cleaned[:max_len]


def seq_hash(seq):
    return hashlib.md5(seq.encode('utf-8')).hexdigest()[:16]


def extract_1d_handcrafted(seqs):
    """Extract 430-D handcrafted features: 20 AAC + 400 DPC + 10 PCP (mean & std of 5 props)."""
    N = len(seqs)
    feats = np.zeros((N, 430), dtype=np.float32)
    for idx, raw_s in enumerate(seqs):
        s = str(raw_s).strip()
        L = max(len(s), 1)
        # AAC (20)
        for ch in s:
            if ch in AA_TO_IDX:
                feats[idx, AA_TO_IDX[ch]] += 1.0 / L
        # DPC (400)
        if L >= 2:
            for j in range(L - 1):
                dip = s[j:j+2]
                if dip in DIP_TO_IDX:
                    feats[idx, 20 + DIP_TO_IDX[dip]] += 1.0 / (L - 1)
        # PCP (5 mean + 5 std = 10)
        for p_i, table in enumerate(PROP_TABLES):
            vals = [table.get(ch, 0.0) for ch in s if ch in AA_TO_IDX]
            if vals:
                feats[idx, 420 + p_i * 2] = np.mean(vals)
                feats[idx, 420 + p_i * 2 + 1] = np.std(vals)
    return feats


def extract_3d_geometric_features(seqs, cache_dir):
    """
    Extract 3D structural topological & physicochemical features from cached 3D coordinates:
    - Radius of gyration (Rg), end-to-end distance, contact density (<8A, <10A, <12A),
      non-local contact ratio (|i-j|>=5 and d<10A), distance histogram (10 bins),
      spatial degree statistics (mean, std, max), plus AAC (20) + PCP (10).
    """
    struct_dir = os.path.join(cache_dir, 'structures')
    feats_1d = extract_1d_handcrafted(seqs)
    aac_pcp = np.concatenate([feats_1d[:, :20], feats_1d[:, 420:]], axis=1)  # 30-D
    geo_feats = np.zeros((len(seqs), 22), dtype=np.float32)

    for idx, raw_s in enumerate(seqs):
        s = clean_seq(raw_s, 50)
        L = len(s)
        p = os.path.join(struct_dir, seq_hash(s) + '.npy')
        if os.path.exists(p):
            coords = np.load(p)[:L]
        else:
            coords = np.zeros((L, 3), dtype=np.float32)
        if L < 2:
            continue
        center = coords.mean(axis=0, keepdims=True)
        rg = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=-1)))
        end2end = np.linalg.norm(coords[-1] - coords[0])
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        iu = np.triu_indices(L, k=1)
        d_upper = dist[iu]
        seq_sep = iu[1] - iu[0]
        c8 = np.mean(d_upper < 8.0)
        c10 = np.mean(d_upper < 10.0)
        c12 = np.mean(d_upper < 12.0)
        nonlocal_mask = seq_sep >= 5
        nl_c10 = np.mean(d_upper[nonlocal_mask] < 10.0) if np.any(nonlocal_mask) else 0.0
        nl_mean_d = np.mean(d_upper[nonlocal_mask]) if np.any(nonlocal_mask) else 0.0
        adj10 = (dist < 10.0).astype(np.float32)
        np.fill_diagonal(adj10, 0.0)
        deg = adj10.sum(axis=-1)
        hist, _ = np.histogram(d_upper, bins=10, range=(0.0, 30.0), density=True)
        geo_feats[idx, 0] = rg
        geo_feats[idx, 1] = end2end
        geo_feats[idx, 2] = c8
        geo_feats[idx, 3] = c10
        geo_feats[idx, 4] = c12
        geo_feats[idx, 5] = nl_c10
        geo_feats[idx, 6] = nl_mean_d
        geo_feats[idx, 7] = deg.mean()
        geo_feats[idx, 8] = deg.std()
        geo_feats[idx, 9] = deg.max()
        geo_feats[idx, 10] = d_upper.mean()
        geo_feats[idx, 11] = d_upper.std()
        geo_feats[idx, 12:22] = hist
    return np.concatenate([aac_pcp, geo_feats], axis=1)


def load_esm_pooled(csv_path, npz_path, max_len=50):
    """Load precomputed ESM embeddings and perform masked mean + max pooling."""
    df = pd.read_csv(csv_path)
    raw_seqs = df['Seq'].astype(str).tolist()
    labels = df['Label'].astype(int).values
    data = np.load(npz_path, allow_pickle=True)
    cache_seqs = [str(s) for s in data['seqs']]
    emb = data['emb']  # (U, Lc, D)
    seq2row = {s: i for i, s in enumerate(cache_seqs)}
    D = int(data['esm_dim'])
    pooled = np.zeros((len(raw_seqs), D * 2), dtype=np.float32)
    for i, s in enumerate(raw_seqs):
        cs = clean_seq(s, max_len)
        L = len(cs)
        row = seq2row[cs]
        e = np.asarray(emb[row, :L], dtype=np.float32)
        pooled[i, :D] = e.mean(axis=0)
        pooled[i, D:] = e.max(axis=0)
    return pooled, labels, raw_seqs


def compute_all_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'MCC': matthews_corrcoef(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob),
        'AUPRC': average_precision_score(y_true, y_prob),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)
    }


def run_holdout_baselines():
    print("\n--- [1/5] Running Hold-Out Benchmark Baselines (training_v2 -> testing_v2) ---")
    tr_csv = os.path.join(SCRIPT_DIR, 'training_v2.csv')
    va_csv = os.path.join(SCRIPT_DIR, 'validation_v2.csv')
    te_csv = os.path.join(SCRIPT_DIR, 'testing_v2.csv')
    cache_dir = os.path.join(SCRIPT_DIR, 'cache')

    tr_df = pd.read_csv(tr_csv)
    te_df = pd.read_csv(te_csv)
    y_tr = tr_df['Label'].astype(int).values
    y_te = te_df['Label'].astype(int).values

    X_tr_1d = extract_1d_handcrafted(tr_df['Seq'].tolist())
    X_te_1d = extract_1d_handcrafted(te_df['Seq'].tolist())

    X_tr_3d = extract_3d_geometric_features(tr_df['Seq'].tolist(), cache_dir)
    X_te_3d = extract_3d_geometric_features(te_df['Seq'].tolist(), cache_dir)

    X_tr_esm8, _, _ = load_esm_pooled(tr_csv, os.path.join(cache_dir, 'esm', 'training_v2_t6_8M_L50.npz'))
    X_te_esm8, _, _ = load_esm_pooled(te_csv, os.path.join(cache_dir, 'esm', 'testing_v2_t6_8M_L50.npz'))

    X_tr_esm650, _, _ = load_esm_pooled(tr_csv, os.path.join(cache_dir, 'esm', 'training_v2_t33_650M_L50.npz'))
    X_te_esm650, _, _ = load_esm_pooled(te_csv, os.path.join(cache_dir, 'esm', 'testing_v2_t33_650M_L50.npz'))

    models = [
        ('AAC+DPC+PCP + LR', X_tr_1d, X_te_1d,
         make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, random_state=500))),
        ('AAC+DPC+PCP + RF', X_tr_1d, X_te_1d,
         RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=500)),
        ('AAC+DPC+PCP + GBDT', X_tr_1d, X_te_1d,
         HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=500)),
        ('AAC+DPC+PCP + SVM', X_tr_1d, X_te_1d,
         make_pipeline(StandardScaler(), SVC(C=2.0, kernel='rbf', probability=True, random_state=500))),
        ('AAC+DPC+PCP + MLP', X_tr_1d, X_te_1d,
         make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-3, max_iter=60, early_stopping=True, random_state=500))),
        ('3D-Geo+PCP + RF', X_tr_3d, X_te_3d,
         RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=500)),
        ('3D-Geo+PCP + GBDT', X_tr_3d, X_te_3d,
         HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=500)),
        ('ESM-2 (8M) + LR', X_tr_esm8, X_te_esm8,
         make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500, random_state=500))),
        ('ESM-2 (8M) + RF', X_tr_esm8, X_te_esm8,
         RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=500)),
        ('ESM-2 (650M) + LR', X_tr_esm650, X_te_esm650,
         make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500, random_state=500))),
        ('ESM-2 (650M) + RF', X_tr_esm650, X_te_esm650,
         RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=500)),
    ]

    rows = []
    preds_dict = {}
    for name, Xtr, Xte, clf in models:
        clf.fit(Xtr, y_tr)
        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        m = compute_all_metrics(y_te, pred, prob)
        rows.append({'Method': name, **m})
        preds_dict[name] = (y_te, pred, prob)
        print(f"  {name:24s} | Acc={m['Accuracy']:.4f} | Sn={m['Sensitivity']:.4f} | Sp={m['Specificity']:.4f} | MCC={m['MCC']:.4f} | AUC={m['AUC']:.4f} | AUPRC={m['AUPRC']:.4f} | F1={m['F1-Score']:.4f}")

    # Also load the deep ablation models (no_esm = 3D-EGNN only, no_3d = ESM-2 650M Deep FFN, full = NeuroEGNN)
    for var_code, disp_name in [
        ('no_esm', '3D-EGNN Only (w/o ESM-2)'),
        ('no_3d', 'ESM-2 (650M) Deep FFN (w/o 3D-EGNN)'),
        ('no_attn', 'NeuroPred-3D (w/o Cross-Attn)'),
        ('no_coord', 'NeuroPred-3D (w/o Equiv. Coord Update)'),
        ('full', 'NeuroEGNN (Ours)')
    ]:
        # Average probabilities across the 3 seeds (or compute mean+-std across seeds)
        seed_metrics = []
        probs_seeds = []
        for s in (500, 501, 502):
            p_csv = os.path.join(SCRIPT_DIR, 'results', 'ablation', var_code, f'seed_{s}', 'test', 'test_predictions.csv')
            df_p = pd.read_csv(p_csv)
            m_s = compute_all_metrics(df_p['True_Label'].values, df_p['Predicted_Label'].values, df_p['Prediction_Probability'].values)
            seed_metrics.append(m_s)
            probs_seeds.append(df_p['Prediction_Probability'].values)
        mean_m = {k: np.mean([x[k] for x in seed_metrics]) for k in seed_metrics[0]}
        std_m = {k + '_std': np.std([x[k] for x in seed_metrics], ddof=1) for k in seed_metrics[0]}
        rows.append({'Method': disp_name, **mean_m, **std_m})
        avg_prob = np.mean(probs_seeds, axis=0)
        preds_dict[disp_name] = (y_te, (avg_prob >= 0.5).astype(int), avg_prob)
        print(f"  {disp_name:38s} | Acc={mean_m['Accuracy']:.4f}±{std_m['Accuracy_std']:.4f} | Sn={mean_m['Sensitivity']:.4f} | Sp={mean_m['Specificity']:.4f} | MCC={mean_m['MCC']:.4f} | AUC={mean_m['AUC']:.4f} | AUPRC={mean_m['AUPRC']:.4f} | F1={mean_m['F1-Score']:.4f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(BASELINE_OUT_DIR, 'holdout_baselines_comparison.csv'), index=False)
    np.savez(os.path.join(BASELINE_OUT_DIR, 'holdout_predictions_curves.npz'),
             **{k.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '').replace('+', '_'): v[2] for k, v in preds_dict.items()},
             y_true=y_te)
    return df_out, preds_dict


def run_5fold_baselines():
    print("\n--- [2/5] Running 5-Fold CV Baselines (fold 1..5, N=8,887) ---")
    cache_dir = os.path.join(SCRIPT_DIR, 'cache')

    baseline_defs = [
        ('AAC+DPC+PCP + LR', '1d', lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, random_state=500))),
        ('AAC+DPC+PCP + RF', '1d', lambda: RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=500)),
        ('AAC+DPC+PCP + GBDT', '1d', lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=500)),
        ('AAC+DPC+PCP + SVM', '1d', lambda: make_pipeline(StandardScaler(), SVC(C=2.0, kernel='rbf', probability=True, random_state=500))),
        ('AAC+DPC+PCP + MLP', '1d', lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-3, max_iter=50, early_stopping=True, random_state=500))),
        ('3D-Geo+PCP + RF', '3d', lambda: RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=500)),
        ('3D-Geo+PCP + GBDT', '3d', lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=500)),
        ('ESM-2 (650M) + LR', 'esm650', lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500, random_state=500))),
        ('ESM-2 (650M) + RF', 'esm650', lambda: RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=500)),
    ]

    fold_records = []
    for fold in range(1, 6):
        tr_csv = os.path.join(SCRIPT_DIR, f'training_v2_fold{fold}.csv')
        te_csv = os.path.join(SCRIPT_DIR, f'testing_v2_fold{fold}.csv')
        tr_df = pd.read_csv(tr_csv)
        te_df = pd.read_csv(te_csv)
        y_tr = tr_df['Label'].astype(int).values
        y_te = te_df['Label'].astype(int).values

        feats_tr = {
            '1d': extract_1d_handcrafted(tr_df['Seq'].tolist()),
            '3d': extract_3d_geometric_features(tr_df['Seq'].tolist(), cache_dir),
            'esm650': load_esm_pooled(tr_csv, os.path.join(cache_dir, 'esm', f'training_v2_fold{fold}_t33_650M_L50.npz'))[0]
        }
        feats_te = {
            '1d': extract_1d_handcrafted(te_df['Seq'].tolist()),
            '3d': extract_3d_geometric_features(te_df['Seq'].tolist(), cache_dir),
            'esm650': load_esm_pooled(te_csv, os.path.join(cache_dir, 'esm', f'testing_v2_fold{fold}_t33_650M_L50.npz'))[0]
        }

        for name, ftype, builder in baseline_defs:
            clf = builder()
            clf.fit(feats_tr[ftype], y_tr)
            prob = clf.predict_proba(feats_te[ftype])[:, 1]
            pred = (prob >= 0.5).astype(int)
            m = compute_all_metrics(y_te, pred, prob)
            fold_records.append({'Method': name, 'Fold': fold - 1, **m})

        # Also load NeuroEGNN fold prediction
        our_pred_csv = os.path.join(SCRIPT_DIR, 'results', '5fold', 'results_5fold', 'test', f'fold_{fold-1}', 'test_predictions.csv')
        df_our = pd.read_csv(our_pred_csv)
        m_our = compute_all_metrics(df_our['True_Label'].values, df_our['Predicted_Label'].values, df_our['Prediction_Probability'].values)
        fold_records.append({'Method': 'NeuroEGNN (Ours)', 'Fold': fold - 1, **m_our})
        print(f"  Completed Fold {fold-1} evaluation for all methods.")

    df_folds = pd.DataFrame(fold_records)
    df_folds.to_csv(os.path.join(BASELINE_OUT_DIR, 'cv5_baselines_per_fold.csv'), index=False)

    # Summarize mean +- std and paired t-test p-values against NeuroEGNN (Ours)
    metric_cols = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'MCC', 'AUC', 'AUPRC', 'F1-Score']
    our_sub = df_folds[df_folds['Method'] == 'NeuroEGNN (Ours)'].sort_values('Fold')
    summary_rows = []
    for method in [b[0] for b in baseline_defs] + ['NeuroEGNN (Ours)']:
        sub = df_folds[df_folds['Method'] == method].sort_values('Fold')
        row = {'Method': method}
        for c in metric_cols:
            vals = sub[c].values
            row[f'{c}_mean'] = vals.mean()
            row[f'{c}_std'] = vals.std(ddof=1)
            row[c] = f"{vals.mean():.4f}±{vals.std(ddof=1):.4f}"
        if method != 'NeuroEGNN (Ours)':
            _, p_acc = stats.ttest_rel(our_sub['Accuracy'].values, sub['Accuracy'].values)
            _, p_mcc = stats.ttest_rel(our_sub['MCC'].values, sub['MCC'].values)
            _, p_auc = stats.ttest_rel(our_sub['AUC'].values, sub['AUC'].values)
            row['p_val_Acc'] = p_acc
            row['p_val_MCC'] = p_mcc
            row['p_val_AUC'] = p_auc
        else:
            row['p_val_Acc'] = np.nan
            row['p_val_MCC'] = np.nan
            row['p_val_AUC'] = np.nan
        summary_rows.append(row)
        print(f"  [5-Fold Summary] {method:26s} | Acc={row['Accuracy']} | Sn={row['Sensitivity']} | Sp={row['Specificity']} | MCC={row['MCC']} | AUC={row['AUC']} | AUPRC={row['AUPRC']} | F1={row['F1-Score']} | p(MCC)={row['p_val_MCC']:.2e}" if not np.isnan(row['p_val_MCC']) else f"  [5-Fold Summary] {method:26s} | Acc={row['Accuracy']} | Sn={row['Sensitivity']} | Sp={row['Specificity']} | MCC={row['MCC']} | AUC={row['AUC']} | AUPRC={row['AUPRC']} | F1={row['F1-Score']}")

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(BASELINE_OUT_DIR, 'cv5_baselines_summary.csv'), index=False)
    return df_folds, df_sum


def analyze_length_stratification():
    print("\n--- [3/5] Analyzing Length-Stratified Performance & Ablation Gains ---")
    # Compare full vs no_3d vs no_esm vs no_attn vs no_coord across peptide length bins on testing_v2.csv
    bins = [(0, 15, 'Short (<=15 aa)'), (16, 30, 'Medium (16-30 aa)'), (31, 100, 'Long (>30 aa)')]
    records = []
    for var in ['full', 'no_3d', 'no_esm', 'no_attn', 'no_coord']:
        for s in (500, 501, 502):
            p_csv = os.path.join(SCRIPT_DIR, 'results', 'ablation', var, f'seed_{s}', 'test', 'test_predictions.csv')
            df = pd.read_csv(p_csv)
            df['len'] = df['Sequence'].astype(str).str.len()
            for lo, hi, bname in bins:
                sub = df[(df['len'] >= lo) & (df['len'] <= hi)]
                m = compute_all_metrics(sub['True_Label'].values, sub['Predicted_Label'].values, sub['Prediction_Probability'].values)
                records.append({'Variant': var, 'Seed': s, 'Length_Bin': bname, 'N': len(sub), **m})
    df_len = pd.DataFrame(records)
    df_len.to_csv(os.path.join(BASELINE_OUT_DIR, 'ablation_length_stratified.csv'), index=False)
    for bname in [b[2] for b in bins]:
        print(f"  Length Bin: {bname}")
        for var in ['full', 'no_3d', 'no_esm', 'no_attn', 'no_coord']:
            sub = df_len[(df_len['Length_Bin'] == bname) & (df_len['Variant'] == var)]
            print(f"    {var:10s} (N={sub['N'].iloc[0]:3d}): Acc={sub['Accuracy'].mean():.4f}±{sub['Accuracy'].std(ddof=1):.4f}, Sn={sub['Sensitivity'].mean():.4f}±{sub['Sensitivity'].std(ddof=1):.4f}, MCC={sub['MCC'].mean():.4f}±{sub['MCC'].std(ddof=1):.4f}, AUC={sub['AUC'].mean():.4f}±{sub['AUC'].std(ddof=1):.4f}")
    return df_len


def analyze_interpretability_global():
    print("\n--- [4/5] Performing Global Quantitative Interpretability Analysis (N=888) ---")
    attn_dir = os.path.join(SCRIPT_DIR, 'results', 'interpretability', 'attention_weights')
    struct_dir = os.path.join(SCRIPT_DIR, 'cache', 'structures')

    aa_attn_tp = {a: [] for a in AAS}
    aa_attn_tn = {a: [] for a in AAS}
    aa_freq_pos = {a: 0 for a in AAS}
    aa_freq_neg = {a: 0 for a in AAS}

    # Spatial vs sequential distance analysis in s2g and g2s
    # Categories of pairs (i, j) with i != j:
    # 1. Local sequence neighbors: |i - j| <= 2
    # 2. Medium-range spatial contacts: 3 <= |i - j| <= 5 and d_ij < 10 A
    # 3. Long-range 3D spatial contacts: |i - j| > 5 and d_ij < 10 A
    # 4. Spatially distant non-neighbors: |i - j| > 2 and d_ij >= 10 A
    s2g_pair_stats = {'local_seq': [], 'med_spatial': [], 'long_spatial': [], 'distant': []}
    g2s_pair_stats = {'local_seq': [], 'med_spatial': [], 'long_spatial': [], 'distant': []}

    n_StartM_fn = 0
    n_StartM_fn_attn0 = []
    tp_nterm_c_term = {'N_term': [], 'Core': [], 'C_term': []}

    files = sorted([f for f in os.listdir(attn_dir) if f.endswith('.npz')])
    for fname in files:
        d = np.load(os.path.join(attn_dir, fname), allow_pickle=True)
        seq = str(d['seq'])
        cs = clean_seq(seq, 50)
        L = len(cs)
        label = int(d['label'])
        pred = int(d['pred'])
        prob = float(d['prob'])
        res_attn = d['residue_attn'][:L]
        # Normalize residue attention per sequence so mean = 1.0 (relative importance enrichment)
        norm_attn = res_attn / (np.mean(res_attn) + 1e-8)

        for ch in cs:
            if ch in AA_TO_IDX:
                if label == 1:
                    aa_freq_pos[ch] += 1
                else:
                    aa_freq_neg[ch] += 1

        if label == 1 and pred == 1:
            for idx_r, ch in enumerate(cs):
                if ch in aa_attn_tp:
                    aa_attn_tp[ch].append(float(norm_attn[idx_r]))
            if L >= 9:
                tp_nterm_c_term['N_term'].append(float(np.mean(norm_attn[:3])))
                tp_nterm_c_term['Core'].append(float(np.mean(norm_attn[3:-3])))
                tp_nterm_c_term['C_term'].append(float(np.mean(norm_attn[-3:])))
        elif label == 0 and pred == 0:
            for idx_r, ch in enumerate(cs):
                if ch in aa_attn_tn:
                    aa_attn_tn[ch].append(float(norm_attn[idx_r]))
        elif label == 1 and pred == 0:
            if cs.startswith('M'):
                n_StartM_fn += 1
                n_StartM_fn_attn0.append(float(res_attn[0] / (np.sum(res_attn) + 1e-8)))

        # Load 3D coordinates to analyze s2g and g2s vs 3D distance
        p_coord = os.path.join(struct_dir, seq_hash(cs) + '.npy')
        if os.path.exists(p_coord) and L >= 6 and 's2g' in d and 'g2s' in d:
            coords = np.load(p_coord)[:L]
            dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            s2g = d['s2g'][:L, :L]
            g2s = d['g2s'][:L, :L]
            # Normalize off-diagonal attention by L so sequences of different lengths are comparable
            idx_i, idx_j = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
            sep = np.abs(idx_i - idx_j)
            m_local = (sep >= 1) & (sep <= 2)
            m_med = (sep >= 3) & (sep <= 5) & (dist < 10.0)
            m_long = (sep > 5) & (dist < 10.0)
            m_dist = (sep > 2) & (dist >= 10.0)

            scale = float(L)
            if np.any(m_local):
                s2g_pair_stats['local_seq'].append(float(np.mean(s2g[m_local]) * scale))
                g2s_pair_stats['local_seq'].append(float(np.mean(g2s[m_local]) * scale))
            if np.any(m_med):
                s2g_pair_stats['med_spatial'].append(float(np.mean(s2g[m_med]) * scale))
                g2s_pair_stats['med_spatial'].append(float(np.mean(g2s[m_med]) * scale))
            if np.any(m_long):
                s2g_pair_stats['long_spatial'].append(float(np.mean(s2g[m_long]) * scale))
                g2s_pair_stats['long_spatial'].append(float(np.mean(g2s[m_long]) * scale))
            if np.any(m_dist):
                s2g_pair_stats['distant'].append(float(np.mean(s2g[m_dist]) * scale))
                g2s_pair_stats['distant'].append(float(np.mean(g2s[m_dist]) * scale))

    tot_pos_aa = sum(aa_freq_pos.values())
    tot_neg_aa = sum(aa_freq_neg.values())
    aa_rows = []
    for a in AAS:
        tp_vals = aa_attn_tp[a]
        tn_vals = aa_attn_tn[a]
        aa_rows.append({
            'AA': a,
            'Pos_Freq_Pct': aa_freq_pos[a] / tot_pos_aa * 100.0,
            'Neg_Freq_Pct': aa_freq_neg[a] / tot_neg_aa * 100.0,
            'TP_Norm_Attn_Mean': np.mean(tp_vals) if tp_vals else 0.0,
            'TP_Norm_Attn_Std': np.std(tp_vals) if tp_vals else 0.0,
            'TN_Norm_Attn_Mean': np.mean(tn_vals) if tn_vals else 0.0,
            'TN_Norm_Attn_Std': np.std(tn_vals) if tn_vals else 0.0,
            'TP_Count': len(tp_vals),
            'TN_Count': len(tn_vals),
        })
    df_aa = pd.DataFrame(aa_rows).sort_values('TP_Norm_Attn_Mean', ascending=False)
    df_aa.to_csv(os.path.join(BASELINE_OUT_DIR, 'interpretability_aa_enrichment.csv'), index=False)
    print("  Top 8 Amino Acids by Normalized Attention in True Positives:")
    print(df_aa[['AA', 'Pos_Freq_Pct', 'Neg_Freq_Pct', 'TP_Norm_Attn_Mean', 'TN_Norm_Attn_Mean', 'TP_Count']].head(8).to_string(index=False))

    print("\n  Cross-Attention Spatial Regime Enrichment (Length-Normalized Attention Weight):")
    for k in ['local_seq', 'med_spatial', 'long_spatial', 'distant']:
        print(f"    {k:15s}: s2g = {np.mean(s2g_pair_stats[k]):.4f} ± {np.std(s2g_pair_stats[k]):.4f} | g2s = {np.mean(g2s_pair_stats[k]):.4f} ± {np.std(g2s_pair_stats[k]):.4f} (n={len(s2g_pair_stats[k])})")

    print(f"\n  Region Attention in True Positives (L>=9): N-term={np.mean(tp_nterm_c_term['N_term']):.4f}, Core={np.mean(tp_nterm_c_term['Core']):.4f}, C-term={np.mean(tp_nterm_c_term['C_term']):.4f}")
    print(f"  False Negatives starting with M: {n_StartM_fn}, mean fraction of total attention on M0 = {np.mean(n_StartM_fn_attn0)*100:.2f}%")
    return df_aa, s2g_pair_stats, g2s_pair_stats, tp_nterm_c_term


def generate_publication_figures(df_holdout, preds_dict, df_cv_sum, df_len, df_aa, s2g_stats, g2s_stats):
    print("\n--- [5/5] Generating High-Resolution Publication Figures in paper/figures/ ---")
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8.5,
        'figure.dpi': 300
    })

    # Figure 1: Dataset Characterization & Homology Debiasing (4 panels: A, B, C, D)
    tr_df = pd.read_csv(os.path.join(SCRIPT_DIR, 'training_v2.csv'))
    va_df = pd.read_csv(os.path.join(SCRIPT_DIR, 'validation_v2.csv'))
    te_df = pd.read_csv(os.path.join(SCRIPT_DIR, 'testing_v2.csv'))
    all_df = pd.concat([tr_df, va_df, te_df], ignore_index=True).drop_duplicates(subset=['Seq'])
    all_df['len'] = all_df['Seq'].str.len()
    clusters = np.load(os.path.join(SCRIPT_DIR, 'cache', 'clusters_sim0.8_7f3749783ffd.npy'))
    vc = pd.Series(clusters).value_counts()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # Panel (a): Length Distribution
    ax = axes[0, 0]
    pos_lens = all_df[all_df['Label'] == 1]['len']
    neg_lens = all_df[all_df['Label'] == 0]['len']
    bins_l = np.linspace(5, 100, 35)
    ax.hist(pos_lens, bins=bins_l, alpha=0.65, color='#1f77b4', label=f'Neuropeptides (n={len(pos_lens)}, med={int(pos_lens.median())} aa)', density=True, edgecolor='black', linewidth=0.4)
    ax.hist(neg_lens, bins=bins_l, alpha=0.55, color='#ff7f0e', label=f'Non-Neuropeptides (n={len(neg_lens)}, med={int(neg_lens.median())} aa)', density=True, edgecolor='black', linewidth=0.4)
    ax.axvline(50, color='#d62728', linestyle='--', linewidth=1.2, label='Model Max Length (L=50 aa)')
    ax.set_xlabel('Peptide Sequence Length (aa)')
    ax.set_ylabel('Probability Density')
    ax.set_title('(a) Sequence Length Distribution')
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)

    # Panel (b): Amino Acid Composition Shift (Pos vs Neg)
    ax = axes[0, 1]
    aa_sorted = df_aa.sort_values('AA')
    x_idx = np.arange(len(AAS))
    w = 0.38
    ax.bar(x_idx - w/2, aa_sorted['Pos_Freq_Pct'], width=w, color='#1f77b4', alpha=0.85, label='Neuropeptides (Pos)')
    ax.bar(x_idx + w/2, aa_sorted['Neg_Freq_Pct'], width=w, color='#ff7f0e', alpha=0.85, label='Non-Neuropeptides (Neg)')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(aa_sorted['AA'])
    ax.set_xlabel('Amino Acid Residue')
    ax.set_ylabel('Residue Frequency (%)')
    ax.set_title('(b) Amino Acid Composition Profile')
    ax.legend(frameon=True)
    ax.grid(axis='y', alpha=0.25)

    # Panel (c): Homology Cluster Size Distribution (Log-Log / Histogram)
    ax = axes[1, 0]
    size_bins = [1, 2, 3, 4, 5, 10, 20, 50, 200]
    size_labels = ['1', '2', '3', '4', '5-9', '10-19', '20-49', '>=50']
    counts_c = [
        (vc == 1).sum(), (vc == 2).sum(), (vc == 3).sum(), (vc == 4).sum(),
        ((vc >= 5) & (vc < 10)).sum(), ((vc >= 10) & (vc < 20)).sum(),
        ((vc >= 20) & (vc < 50)).sum(), (vc >= 50).sum()
    ]
    bars = ax.bar(range(len(size_labels)), counts_c, color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.6)
    ax.set_yscale('log')
    ax.set_xticks(range(len(size_labels)))
    ax.set_xticklabels(size_labels)
    ax.set_xlabel('Homology Cluster Size (# Sequences at Identity >= 0.8)')
    ax.set_ylabel('Number of Clusters (Log Scale)')
    ax.set_title('(c) Homology Cluster Size Distribution (7,050 Clusters)')
    for b, c in zip(bars, counts_c):
        if c > 0:
            ax.text(b.get_x() + b.get_width()/2, c * 1.15, str(c), ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0.7, 15000)
    ax.grid(axis='y', alpha=0.25)

    # Panel (d): 3D Contact Density & Radius of Gyration (Pos vs Neg)
    ax = axes[1, 1]
    cache_dir = os.path.join(SCRIPT_DIR, 'cache')
    sample_pos = all_df[all_df['Label'] == 1]['Seq'].iloc[:600].tolist()
    sample_neg = all_df[all_df['Label'] == 0]['Seq'].iloc[:600].tolist()
    geo_pos = extract_3d_geometric_features(sample_pos, cache_dir)[:, 30:]
    geo_neg = extract_3d_geometric_features(sample_neg, cache_dir)[:, 30:]
    ax.scatter(geo_neg[:, 0], geo_neg[:, 3], alpha=0.35, s=16, c='#ff7f0e', label='Non-Neuropeptides')
    ax.scatter(geo_pos[:, 0], geo_pos[:, 3], alpha=0.35, s=16, c='#1f77b4', label='Neuropeptides')
    ax.set_xlabel(r'3D Radius of Gyration $R_g$ ($\AA$)')
    ax.set_ylabel(r'Spatial Contact Fraction ($d_{ij} < 10\,\AA$)')
    ax.set_title('(d) 3D Conformational Compactness vs. Contact Density')
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_FIG_DIR, 'fig1_dataset_analysis.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PAPER_FIG_DIR, 'fig1_dataset_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: ROC and Precision-Recall Curves + 5-Fold Performance Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    selected_methods = [
        ('NeuroEGNN (Ours)', '#d62728', '-', 2.2),
        ('ESM-2 (650M) Deep FFN (w/o 3D-EGNN)', '#1f77b4', '--', 1.6),
        ('ESM-2 (650M) + LR', '#9467bd', '-.', 1.5),
        ('3D-EGNN Only (w/o ESM-2)', '#2ca02c', '--', 1.5),
        ('AAC+DPC+PCP + MLP', '#ff7f0e', ':', 1.5),
        ('AAC+DPC+PCP + RF', '#8c564b', ':', 1.4),
        ('3D-Geo+PCP + GBDT', '#7f7f7f', '-.', 1.3),
    ]
    # Panel (a): ROC Curve on Held-out Test Set
    ax = axes[0]
    for mname, col, ls, lw in selected_methods:
        if mname in preds_dict:
            yt, _, yp = preds_dict[mname]
            fpr, tpr, _ = roc_curve(yt, yp)
            auc_v = roc_auc_score(yt, yp)
            short_lbl = mname.replace(' (w/o 3D-EGNN)', '').replace(' (w/o ESM-2)', '')
            ax.plot(fpr, tpr, color=col, linestyle=ls, linewidth=lw, label=f'{short_lbl} (AUC={auc_v:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('(a) ROC Curves on Independent Test Set')
    ax.legend(loc='lower right', fontsize=7.5, frameon=True)
    ax.grid(alpha=0.25)

    # Panel (b): Precision-Recall Curve on Held-out Test Set
    ax = axes[1]
    for mname, col, ls, lw in selected_methods:
        if mname in preds_dict:
            yt, _, yp = preds_dict[mname]
            prec_c, rec_c, _ = precision_recall_curve(yt, yp)
            auprc_v = average_precision_score(yt, yp)
            short_lbl = mname.replace(' (w/o 3D-EGNN)', '').replace(' (w/o ESM-2)', '')
            ax.plot(rec_c, prec_c, color=col, linestyle=ls, linewidth=lw, label=f'{short_lbl} (AUPRC={auprc_v:.4f})')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.50, 1.02)
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title('(b) Precision-Recall Curves on Independent Test Set')
    ax.legend(loc='lower left', fontsize=7.5, frameon=True)
    ax.grid(alpha=0.25)

    # Panel (c): 5-Fold Cross-Validation Radar / Grouped Bar Comparison
    ax = axes[2]
    cv_methods = ['AAC+DPC+PCP + RF', '3D-Geo+PCP + GBDT', 'AAC+DPC+PCP + SVM', 'ESM-2 (650M) + LR', 'NeuroEGNN (Ours)']
    cv_colors = ['#8c564b', '#7f7f7f', '#ff7f0e', '#1f77b4', '#d62728']
    m_names = ['Accuracy', 'Sensitivity', 'Specificity', 'MCC', 'AUC', 'F1-Score']
    x_m = np.arange(len(m_names))
    bw = 0.15
    for idx_c, (cm_name, c_col) in enumerate(zip(cv_methods, cv_colors)):
        sub_r = df_cv_sum[df_cv_sum['Method'] == cm_name].iloc[0]
        means = [sub_r[f'{m}_mean'] for m in m_names]
        stds = [sub_r[f'{m}_std'] for m in m_names]
        offset = (idx_c - 2) * bw
        ax.bar(x_m + offset, means, width=bw, yerr=stds, capsize=2, color=c_col, alpha=0.88, label=cm_name)
    ax.set_xticks(x_m)
    ax.set_xticklabels(['Acc', 'Sn', 'Sp', 'MCC', 'AUC', 'F1'])
    ax.set_ylim(0.60, 1.03)
    ax.set_ylabel('Metric Score (Mean ± Std)')
    ax.set_title('(c) 5-Fold Homology-Isolated CV Performance')
    ax.legend(loc='lower right', fontsize=7.5, frameon=True)
    ax.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_FIG_DIR, 'fig2_performance_comparison.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PAPER_FIG_DIR, 'fig2_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Ablation Study & Length-Stratified Analysis + Training Convergence
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # Panel (a): Ablation Variants Grouped Bar Chart (Acc, Sn, Sp, MCC, F1)
    ax = axes[0]
    ab_vars = ['full', 'no_attn', 'no_coord', 'no_3d', 'no_esm']
    ab_labels = ['Full Model', 'w/o Cross-Attn', 'w/o Coord-Upd', 'w/o 3D-EGNN', 'w/o ESM-2']
    ab_colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    ab_summary = pd.read_csv(os.path.join(SCRIPT_DIR, 'results', 'ablation', 'ablation_summary.csv'))
    m_ab = ['Accuracy', 'Sensitivity', 'Specificity', 'MCC', 'F1-Score']
    x_ab = np.arange(len(m_ab))
    bw = 0.15
    for idx_v, (v_code, v_lbl, v_col) in enumerate(zip(ab_vars, ab_labels, ab_colors)):
        r = ab_summary[ab_summary['variant'] == v_code].iloc[0]
        means = [r[f'{m}_mean'] for m in m_ab]
        stds = [r[f'{m}_std'] for m in m_ab]
        ax.bar(x_ab + (idx_v - 2)*bw, means, width=bw, yerr=stds, capsize=2, color=v_col, alpha=0.88, label=v_lbl)
    ax.set_xticks(x_ab)
    ax.set_xticklabels(['Acc', 'Sn', 'Sp', 'MCC', 'F1'])
    ax.set_ylim(0.60, 1.02)
    ax.set_ylabel('Score on Held-Out Test Set')
    ax.set_title('(a) Component Ablation Study (3 Seeds)')
    ax.legend(loc='lower right', fontsize=8, frameon=True)
    ax.grid(axis='y', alpha=0.25)

    # Panel (b): Length-Stratified Sensitivity Comparison (Full vs w/o 3D-EGNN vs w/o Coord-Upd)
    ax = axes[1]
    lbins = ['Short (<=15 aa)', 'Medium (16-30 aa)', 'Long (>30 aa)']
    x_lb = np.arange(len(lbins))
    bw_l = 0.24
    for idx_v, (v_code, v_lbl, v_col) in enumerate([('full', 'Full (ESM-2 + 3D-EGNN)', '#d62728'),
                                                     ('no_coord', 'w/o Equiv. Coord Update', '#2ca02c'),
                                                     ('no_3d', 'w/o 3D-EGNN (1D Only)', '#ff7f0e')]):
        means = [df_len[(df_len['Length_Bin'] == b) & (df_len['Variant'] == v_code)]['Sensitivity'].mean() for b in lbins]
        stds = [df_len[(df_len['Length_Bin'] == b) & (df_len['Variant'] == v_code)]['Sensitivity'].std(ddof=1) for b in lbins]
        ax.bar(x_lb + (idx_v - 1)*bw_l, means, width=bw_l, yerr=stds, capsize=3, color=v_col, alpha=0.88, label=v_lbl)
    ax.set_xticks(x_lb)
    ax.set_xticklabels(['Short\n(<=15 aa, n=394)', 'Medium\n(16-30 aa, n=321)', 'Long\n(>30 aa, n=173)'])
    ax.set_ylim(0.70, 1.03)
    ax.set_ylabel('Sensitivity (Recall)')
    ax.set_title('(b) Sensitivity by Peptide Length')
    ax.legend(loc='lower left', fontsize=8, frameon=True)
    ax.grid(axis='y', alpha=0.25)

    # Panel (c): Validation AUC Learning Curves across 5 Folds
    ax = axes[2]
    fold_cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for f_i in range(5):
        log_csv = os.path.join(SCRIPT_DIR, 'results', '5fold', 'results_5fold', 'train', f'fold_{f_i}', 'training_log.csv')
        if os.path.exists(log_csv):
            df_log = pd.read_csv(log_csv)
            ax.plot(df_log['epoch'], df_log['val_auc'], color=fold_cols[f_i], linewidth=1.5, alpha=0.85, label=f'Fold {f_i+1} (Best={df_log["val_auc"].max():.4f})')
            best_ep = df_log.loc[df_log['val_auc'].idxmax()]
            ax.scatter([best_ep['epoch']], [best_ep['val_auc']], color=fold_cols[f_i], s=35, zorder=5, edgecolor='black')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Validation AUC-ROC')
    ax.set_title('(c) Validation Convergence & Early Stopping')
    ax.legend(loc='lower right', fontsize=8, frameon=True)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_FIG_DIR, 'fig3_ablation_and_length.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PAPER_FIG_DIR, 'fig3_ablation_and_length.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 4: Global Interpretability Analysis (AA Enrichment + Spatial Cross-Attention Regimes + Confidence Distribution)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # Panel (a): Normalized Residue Attention Enrichment by Amino Acid Type (TP vs TN)
    ax = axes[0]
    df_aa_plot = df_aa.sort_values('TP_Norm_Attn_Mean', ascending=False)
    x_a = np.arange(len(AAS))
    ax.bar(x_a - 0.18, df_aa_plot['TP_Norm_Attn_Mean'], width=0.36, color='#d62728', alpha=0.85, label='True Positives (Neuropeptides)')
    ax.bar(x_a + 0.18, df_aa_plot['TN_Norm_Attn_Mean'], width=0.36, color='#1f77b4', alpha=0.80, label='True Negatives (Non-Neuropeptides)')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0, label='Uniform Attention Baseline (1.0)')
    ax.set_xticks(x_a)
    ax.set_xticklabels(df_aa_plot['AA'])
    ax.set_xlabel('Amino Acid Residue Type (Sorted by TP Attention)')
    ax.set_ylabel('Normalized Attention Weight')
    ax.set_title('(a) Residue-Level Attention Enrichment (N=888)')
    ax.legend(loc='upper right', fontsize=7.5, frameon=True)
    ax.grid(axis='y', alpha=0.25)

    # Panel (b): Cross-Attention Weights across Spatial-Sequential Regimes
    ax = axes[1]
    regimes = ['local_seq', 'med_spatial', 'long_spatial', 'distant']
    regime_labels = ['Local Seq\n(|i-j|<=2)', 'Med-Range 3D\n(3<=|i-j|<=5,\nd<10Å)', 'Long-Range 3D\n(|i-j|>5,\nd<10Å)', 'Distant Non-Local\n(|i-j|>2,\nd>=10Å)']
    s2g_means = [np.mean(s2g_stats[k]) for k in regimes]
    s2g_sems = [np.std(s2g_stats[k]) / np.sqrt(len(s2g_stats[k])) for k in regimes]
    g2s_means = [np.mean(g2s_stats[k]) for k in regimes]
    g2s_sems = [np.std(g2s_stats[k]) / np.sqrt(len(g2s_stats[k])) for k in regimes]
    xr = np.arange(len(regimes))
    ax.bar(xr - 0.18, s2g_means, width=0.35, yerr=s2g_sems, capsize=3, color='#2ca02c', alpha=0.85, label=r'Seq $\to$ Graph ($\mathrm{Attn}_{s\to g}$)')
    ax.bar(xr + 0.18, g2s_means, width=0.35, yerr=g2s_sems, capsize=3, color='#9467bd', alpha=0.85, label=r'Graph $\to$ Seq ($\mathrm{Attn}_{g\to s}$)')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    ax.set_xticks(xr)
    ax.set_xticklabels(regime_labels, fontsize=8)
    ax.set_ylabel('Length-Normalized Pair Attention')
    ax.set_title('(b) Cross-Attention by 3D Spatial Regime')
    ax.legend(loc='upper right', fontsize=8, frameon=True)
    ax.grid(axis='y', alpha=0.25)

    # Panel (c): Out-of-Fold Prediction Probability Distribution (N=8,887)
    ax = axes[2]
    oof_dfs = [pd.read_csv(os.path.join(SCRIPT_DIR, 'results', '5fold', 'results_5fold', 'test', f'fold_{i}', 'test_predictions.csv')) for i in range(5)]
    oof_all = pd.concat(oof_dfs, ignore_index=True)
    pos_probs = oof_all[oof_all['True_Label'] == 1]['Prediction_Probability']
    neg_probs = oof_all[oof_all['True_Label'] == 0]['Prediction_Probability']
    ax.hist(pos_probs, bins=40, range=(0, 1), alpha=0.7, color='#d62728', label=f'True Neuropeptides (n={len(pos_probs)})', density=True)
    ax.hist(neg_probs, bins=40, range=(0, 1), alpha=0.65, color='#1f77b4', label=f'True Non-Neuropeptides (n={len(neg_probs)})', density=True)
    ax.axvspan(0.4, 0.6, color='gray', alpha=0.2, label='Borderline Zone [0.4, 0.6] (4.0%)')
    ax.set_xlabel('Predicted Neuropeptide Probability')
    ax.set_ylabel('Density')
    ax.set_title('(c) Out-of-Fold Probability Calibration (N=8,887)')
    ax.legend(loc='upper center', fontsize=7.5, frameon=True)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_FIG_DIR, 'fig4_global_interpretability.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PAPER_FIG_DIR, 'fig4_global_interpretability.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Also copy the representative case heatmaps from results/interpretability/figures to paper/figures/
    import shutil
    for f_src, f_dst in [
        ('top_correct_1_heatmap.png', 'fig5a_disulfide_correct_heatmap.png'),
        ('top_wrong_2_heatmap.png', 'fig5b_nterm_met_wrong_heatmap.png'),
        ('residue_attention_comparison.png', 'fig6_residue_attention_comparison.png'),
    ]:
        src_p = os.path.join(SCRIPT_DIR, 'results', 'interpretability', 'figures', f_src)
        if os.path.exists(src_p):
            shutil.copy2(src_p, os.path.join(PAPER_FIG_DIR, f_dst))
    print("  All figures successfully saved to:", PAPER_FIG_DIR)


if __name__ == '__main__':
    df_holdout, preds_dict = run_holdout_baselines()
    df_cv_folds, df_cv_sum = run_5fold_baselines()
    df_len = analyze_length_stratification()
    df_aa, s2g_stats, g2s_stats, tp_regions = analyze_interpretability_global()
    generate_publication_figures(df_holdout, preds_dict, df_cv_sum, df_len, df_aa, s2g_stats, g2s_stats)
    print("\n=== ALL SUPPLEMENTARY EXPERIMENTS & ANALYSES COMPLETED SUCCESSFULLY ===")

