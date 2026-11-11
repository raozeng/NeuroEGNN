# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from run_jcim_supplements import (
    AAS, AA_TO_IDX, clean_seq, seq_hash,
    load_ckpt, load_batch_tensors, forward_neuroegnn_numpy,
    extract_plm_multiscale_conv_features, extract_multiscale_seq_features,
    compute_metrics
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def delong_roc_variance(ground_truth, predictions):
    """Fast DeLong AUC variance & covariance for paired two-sample test (Sun & Xu 2014)."""
    ground_truth = np.asarray(ground_truth, dtype=int)
    pos = predictions[:, ground_truth == 1]
    neg = predictions[:, ground_truth == 0]
    m = pos.shape[1]
    n = neg.shape[1]
    k = predictions.shape[0]

    tx = np.zeros((k, m), dtype=np.float64)
    ty = np.zeros((k, n), dtype=np.float64)
    aucs = np.zeros(k, dtype=np.float64)
    for r in range(k):
        # midrank comparison: pos vs neg
        p = pos[r][:, None]
        q = neg[r][None, :]
        cmp = (p > q).astype(np.float64) + 0.5 * (p == q).astype(np.float64)
        aucs[r] = cmp.mean()
        tx[r] = cmp.mean(axis=1)
        ty[r] = cmp.mean(axis=0)

    sx = np.cov(tx)
    sy = np.cov(ty)
    s = sx / m + sy / n
    return aucs, s


def delong_p_value(y_true, prob1, prob2):
    preds = np.vstack([prob1, prob2])
    aucs, cov = delong_roc_variance(y_true, preds)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 1e-16:
        return 1.0, diff
    z = diff / np.sqrt(var)
    p = 2.0 * stats.norm.sf(abs(z))
    return float(p), float(diff)


def mcnemar_p_value(y_true, pred1, pred2):
    y_true = np.asarray(y_true, dtype=int)
    c1 = (np.asarray(pred1, dtype=int) == y_true)
    c2 = (np.asarray(pred2, dtype=int) == y_true)
    b = int(np.sum(c1 & (~c2)))  # model 1 right, model 2 wrong
    c = int(np.sum((~c1) & c2))  # model 1 wrong, model 2 right
    if b + c == 0:
        return 1.0, b, c
    res = stats.binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided")
    return float(res.pvalue), b, c


def main():
    out = {}

    # 1. Exact Dataset & Split Verification
    tr_df = pd.read_csv(os.path.join(CODE_DIR, "training_v2.csv"))
    va_df = pd.read_csv(os.path.join(CODE_DIR, "validation_v2.csv"))
    te_df = pd.read_csv(os.path.join(CODE_DIR, "testing_v2.csv"))
    all_df = pd.concat([tr_df, va_df, te_df], ignore_index=True)
    assert len(all_df) == 8887 and all_df["Seq"].nunique() == 8887

    def split_stats(df):
        seqs = df["Seq"].astype(str).str.strip()
        lens = seqs.str.len()
        pos = df[df["Label"] == 1]
        neg = df[df["Label"] == 0]
        p_seqs = pos["Seq"].astype(str).str.strip()
        n_seqs = neg["Seq"].astype(str).str.strip()
        return {
            "N": int(len(df)),
            "pos": int(len(pos)),
            "neg": int(len(neg)),
            "pos_ratio": float(len(pos) / len(df) * 100),
            "len_min": int(lens.min()),
            "len_max": int(lens.max()),
            "len_mean": float(lens.mean()),
            "len_sd": float(lens.std(ddof=1)),
            "le50_count": int((lens <= 50).sum()),
            "le50_pct": float((lens <= 50).mean() * 100),
            "pos_m1_cnt": int(p_seqs.str.startswith("M").sum()),
            "pos_m1_pct": float(p_seqs.str.startswith("M").mean() * 100),
            "neg_m1_cnt": int(n_seqs.str.startswith("M").sum()),
            "neg_m1_pct": float(n_seqs.str.startswith("M").mean() * 100),
            "non_m1_cnt": int((~seqs.str.startswith("M")).sum()),
            "non_m1_pos": int((~p_seqs.str.startswith("M")).sum()),
            "non_m1_neg": int((~n_seqs.str.startswith("M")).sum()),
        }

    ds_stats = {
        "all_data": split_stats(all_df),
        "training_v2": split_stats(tr_df),
        "validation_v2": split_stats(va_df),
        "testing_v2": split_stats(te_df),
    }
    for f in range(1, 6):
        fdf = pd.read_csv(os.path.join(CODE_DIR, f"testing_v2_fold{f}.csv"))
        ds_stats[f"testing_v2_fold{f}"] = split_stats(fdf)
    out["dataset_stats"] = ds_stats
    print("=== Dataset Split Stats ===")
    for k, v in ds_stats.items():
        print(f"  {k:18s}: N={v['N']} ({v['pos']}+/{v['neg']}-), L={v['len_min']}-{v['len_max']} ({v['len_mean']:.2f}±{v['len_sd']:.2f}), "
              f"<=50aa={v['le50_pct']:.2f}%, M1={v['pos_m1_cnt']}({v['pos_m1_pct']:.2f}%)/{v['neg_m1_cnt']}({v['neg_m1_pct']:.2f}%), "
              f"NonM1={v['non_m1_cnt']} ({v['non_m1_pos']}+/{v['non_m1_neg']}-)")

    # Check length bins in testing_v2.csv (raw length vs cleaned<=50 length)
    te_raw_lens = te_df["Seq"].astype(str).str.strip().str.len()
    print("\n=== testing_v2 Length Bins ===")
    for lo, hi, name in [(0, 15, "<=15"), (16, 30, "16-30"), (31, 200, ">30")]:
        m = (te_raw_lens >= lo) & (te_raw_lens <= hi)
        sub = te_df[m]
        print(f"  Raw {name:6s}: n={len(sub)} (Pos={int((sub['Label']==1).sum())}, Neg={int((sub['Label']==0).sum())})")

    # Check exact parameter counts from checkpoints
    param_counts = {}
    for var in ["full", "no_coord", "no_attn", "no_3d", "no_esm"]:
        ck = load_ckpt(os.path.join(CODE_DIR, "results", "ablation", var, "seed_500", "train", "best_model.pth"))
        n_p = sum(int(np.prod(arr.shape)) for arr in ck["state_dict"].values())
        param_counts[var] = n_p
    out["param_counts"] = param_counts
    print("\n=== Exact Trainable Parameter Counts from State Dict ===")
    print(param_counts)

    # 2. Clean Zero-Leakage Interpretability on testing_v2.csv using ablation/full/seed_500..502
    test_csv = os.path.join(CODE_DIR, "testing_v2.csv")
    test_npz = os.path.join(CODE_DIR, "cache", "esm", "testing_v2_t33_650M_L50.npz")
    batch_dict = load_batch_tensors(test_csv, test_npz)
    y_te = batch_dict["labels"]
    lengths = batch_dict["lengths"]
    cleaned_seqs = batch_dict["cleaned_seqs"]
    raw_seqs = batch_dict["raw_seqs"]

    seed_outs = {}
    for s in (500, 501, 502):
        ck = load_ckpt(os.path.join(CODE_DIR, "results", "ablation", "full", f"seed_{s}", "train", "best_model.pth"))
        seed_outs[s] = forward_neuroegnn_numpy(ck["state_dict"], ck["config"], batch_dict, return_details=True)

    # Ensemble average across 3 clean holdout seeds + check seed_500 individually
    for mode_name, probs_arr, s2g_arr, g2s_arr in [
        ("seed_500", seed_outs[500]["probs"], seed_outs[500]["attn_s2g"], seed_outs[500]["attn_g2s"]),
        ("ens_3seeds",
         np.mean([seed_outs[s]["probs"] for s in (500, 501, 502)], axis=0),
         np.mean([seed_outs[s]["attn_s2g"] for s in (500, 501, 502)], axis=0),
         np.mean([seed_outs[s]["attn_g2s"] for s in (500, 501, 502)], axis=0)),
    ]:
        preds_arr = (probs_arr >= 0.5).astype(int)
        s2g_pair_stats = {"local_seq": [], "med_spatial": [], "long_spatial": [], "distant": []}
        g2s_pair_stats = {"local_seq": [], "med_spatial": [], "long_spatial": [], "distant": []}
        aa_tp = {a: [] for a in AAS}
        aa_tn = {a: [] for a in AAS}
        aa_tp_internal = {a: [] for a in AAS}
        aa_tn_internal = {a: [] for a in AAS}
        m1_tp_vals, m1_tn_vals = [], []
        tp_seg = {"N_term": [], "Core": [], "C_term": []}
        fn_m1_fracs = []
        rel_pos_tp = [[] for _ in range(20)]
        rel_pos_tn = [[] for _ in range(20)]

        for i in range(len(y_te)):
            L = int(lengths[i])
            cs = cleaned_seqs[i]
            y = int(y_te[i])
            pred = int(preds_arr[i])
            s2g = s2g_arr[i, :L, :L]
            g2s = g2s_arr[i, :L, :L]
            # How did run_interpretability.py define residue_attn? Let's check:
            # Incoming marginal attention (mean over query axis=0 of 0.5*(s2g + g2s))
            res_attn = 0.5 * (s2g.mean(axis=0) + g2s.mean(axis=0))
            norm_attn = res_attn / (np.mean(res_attn) + 1e-8)

            if y == 1 and pred == 1:
                for r_i, ch in enumerate(cs):
                    if ch in aa_tp:
                        aa_tp[ch].append(float(norm_attn[r_i]))
                        if r_i >= 1:
                            aa_tp_internal[ch].append(float(norm_attn[r_i]))
                        elif ch == "M":
                            m1_tp_vals.append(float(norm_attn[r_i]))
                    b_idx = min(19, int(np.floor(r_i / max(L, 1) * 20)))
                    rel_pos_tp[b_idx].append(float(norm_attn[r_i]))
                if L >= 9:
                    k_seg = min(3, L // 3)
                    tp_seg["N_term"].append(float(np.mean(norm_attn[:k_seg])))
                    tp_seg["Core"].append(float(np.mean(norm_attn[k_seg:-k_seg])))
                    tp_seg["C_term"].append(float(np.mean(norm_attn[-k_seg:])))
            elif y == 0 and pred == 0:
                for r_i, ch in enumerate(cs):
                    if ch in aa_tn:
                        aa_tn[ch].append(float(norm_attn[r_i]))
                        if r_i >= 1:
                            aa_tn_internal[ch].append(float(norm_attn[r_i]))
                        elif ch == "M":
                            m1_tn_vals.append(float(norm_attn[r_i]))
                    b_idx = min(19, int(np.floor(r_i / max(L, 1) * 20)))
                    rel_pos_tn[b_idx].append(float(norm_attn[r_i]))
            elif y == 1 and pred == 0 and cs.startswith("M"):
                fn_m1_fracs.append(float(res_attn[0] / (np.sum(res_attn) + 1e-8)))

            if L >= 6:
                coords = batch_dict["coords"][i, :L]
                dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
                idx_i, idx_j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
                sep = np.abs(idx_i - idx_j)
                m_local = (sep >= 1) & (sep <= 2)
                m_med = (sep >= 3) & (sep <= 5) & (dist < 10.0)
                m_long = (sep > 5) & (dist < 10.0)
                m_dist = (sep > 2) & (dist >= 10.0)
                scale = float(L)
                for key, mask_k in [("local_seq", m_local), ("med_spatial", m_med), ("long_spatial", m_long), ("distant", m_dist)]:
                    if np.any(mask_k):
                        s2g_pair_stats[key].append(float(np.mean(s2g[mask_k]) * scale))
                        g2s_pair_stats[key].append(float(np.mean(g2s[mask_k]) * scale))

        print(f"\n=== Clean Zero-Leakage Interpretability ({mode_name}) ===")
        for k in ["local_seq", "med_spatial", "long_spatial", "distant"]:
            print(f"  {k:14s}: s2g={np.mean(s2g_pair_stats[k]):.4f}±{np.std(s2g_pair_stats[k]):.4f} | "
                  f"g2s={np.mean(g2s_pair_stats[k]):.4f}±{np.std(g2s_pair_stats[k]):.4f} (n={len(s2g_pair_stats[k])})")
        lr_gain = (np.mean(s2g_pair_stats["long_spatial"]) / np.mean(s2g_pair_stats["local_seq"]) - 1.0) * 100
        print(f"  Long-range vs Local s2g enrichment: +{lr_gain:.2f}%")
        print(f"  TP Segments (L>=9, n={len(tp_seg['N_term'])}): "
              f"N={np.mean(tp_seg['N_term']):.4f}±{np.std(tp_seg['N_term']):.4f}, "
              f"Core={np.mean(tp_seg['Core']):.4f}±{np.std(tp_seg['Core']):.4f}, "
              f"C={np.mean(tp_seg['C_term']):.4f}±{np.std(tp_seg['C_term']):.4f} "
              f"(C/Core={(np.mean(tp_seg['C_term'])/np.mean(tp_seg['Core'])-1)*100:+.2f}%)")
        print("  AA TP vs TN Normalized Attention:")
        aa_summary = []
        for a in AAS:
            aa_summary.append((a, np.mean(aa_tp[a]), np.mean(aa_tn[a]), len(aa_tp[a]), len(aa_tn[a]),
                               np.mean(aa_tp_internal[a]) if aa_tp_internal[a] else 0.0))
        aa_summary.sort(key=lambda x: x[1], reverse=True)
        for a, tp_m, tn_m, n_tp, n_tn, tp_int in aa_summary[:10]:
            print(f"    {a}: TP={tp_m:.4f} (internal={tp_int:.4f}, n={n_tp}) | TN={tn_m:.4f} (n={n_tn}) | TP/TN={tp_m/tn_m:.2f}x")
        print(f"  Met1 specifically: TP_Met1={m1_tp_vals}, TN_Met1 mean={np.mean(m1_tn_vals):.4f} (n={len(m1_tn_vals)})")
        print(f"  FN starting with M (n={len(fn_m1_fracs)}): mean M1 attn share = {np.mean(fn_m1_fracs)*100:.2f}%")

        # Case #608 and #156
        for cid in [608, 156]:
            L_c = int(lengths[cid])
            cs_c = cleaned_seqs[cid]
            s2g_c = s2g_arr[cid, :L_c, :L_c]
            g2s_c = g2s_arr[cid, :L_c, :L_c]
            res_c = 0.5 * (s2g_c.mean(axis=0) + g2s_c.mean(axis=0))
            norm_c = res_c / np.mean(res_c)
            if cid == 608:
                cys_idx = [j for j, ch in enumerate(cs_c) if ch == "C"]
                print(f"  Case #608 ({cs_c}): P={probs_arr[cid]:.4f}, Cys norm attn={np.mean(norm_c[cys_idx]):.2f}x")
            if cid == 156:
                print(f"  Case #156 ({cs_c}): P={probs_arr[cid]:.4f}, M1 share={res_c[0]/np.sum(res_c)*100:.2f}% "
                      f"(s2g M1 share={s2g_c.mean(axis=0)[0]/np.sum(s2g_c.mean(axis=0))*100:.2f}%, "
                      f"g2s M1 share={g2s_c.mean(axis=0)[0]/np.sum(g2s_c.mean(axis=0))*100:.2f}%)")

        out[f"interp_{mode_name}"] = {
            "s2g_local": [float(np.mean(s2g_pair_stats["local_seq"])), float(np.std(s2g_pair_stats["local_seq"]))],
            "s2g_med": [float(np.mean(s2g_pair_stats["med_spatial"])), float(np.std(s2g_pair_stats["med_spatial"]))],
            "s2g_long": [float(np.mean(s2g_pair_stats["long_spatial"])), float(np.std(s2g_pair_stats["long_spatial"]))],
            "s2g_dist": [float(np.mean(s2g_pair_stats["distant"])), float(np.std(s2g_pair_stats["distant"]))],
            "g2s_local": [float(np.mean(g2s_pair_stats["local_seq"])), float(np.std(g2s_pair_stats["local_seq"]))],
            "g2s_med": [float(np.mean(g2s_pair_stats["med_spatial"])), float(np.std(g2s_pair_stats["med_spatial"]))],
            "g2s_long": [float(np.mean(g2s_pair_stats["long_spatial"])), float(np.std(g2s_pair_stats["long_spatial"]))],
            "g2s_dist": [float(np.mean(g2s_pair_stats["distant"])), float(np.std(g2s_pair_stats["distant"]))],
            "tp_N": [float(np.mean(tp_seg["N_term"])), float(np.std(tp_seg["N_term"]))],
            "tp_Core": [float(np.mean(tp_seg["Core"])), float(np.std(tp_seg["Core"]))],
            "tp_C": [float(np.mean(tp_seg["C_term"])), float(np.std(tp_seg["C_term"]))],
            "tp_L9_n": len(tp_seg["N_term"]),
            "fn_m1_n": len(fn_m1_fracs),
            "fn_m1_share_pct": float(np.mean(fn_m1_fracs) * 100),
            "p_608": float(probs_arr[608]),
            "p_156": float(probs_arr[156]),
        }

    # 3. Rg-normalized RMSD Mean, SD, Median, IQR across layers on testing_v2.csv (seed_500)
    coord_traj = seed_outs[500]["coord_traj"]
    print("\n=== Rg-Normalized RMSD Distribution Across Layers (seed_500) ===")
    rmsd_dist = {}
    for l in range(5):
        vals = []
        for i in range(len(y_te)):
            L = int(lengths[i])
            if L < 9:
                continue
            x0 = coord_traj[0][i, :L] - coord_traj[0][i, :L].mean(axis=0, keepdims=True)
            xl = coord_traj[l][i, :L] - coord_traj[l][i, :L].mean(axis=0, keepdims=True)
            rg0 = np.sqrt(np.mean(np.sum(x0 ** 2, axis=-1))) + 1e-8
            rgl = np.sqrt(np.mean(np.sum(xl ** 2, axis=-1))) + 1e-8
            xl_sc = xl * (rg0 / rgl)
            vals.append(float(np.sqrt(np.mean(np.sum((xl_sc - x0) ** 2, axis=-1)))))
        arr = np.array(vals)
        q25, q50, q75, q95 = np.percentile(arr, [25, 50, 75, 95])
        rmsd_dist[l] = {
            "mean": float(arr.mean()), "sd": float(arr.std()),
            "median": float(q50), "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25), "q95": float(q95)
        }
        print(f"  Layer {l}: Mean={arr.mean():.3f} ± {arr.std():.3f} A | Median={q50:.3f} [IQR {q25:.3f}-{q75:.3f}, 95th={q95:.3f}] A")
    out["rmsd_rg_dist"] = rmsd_dist

    # 4. Run NeuroPred-FRL on 5-Fold CV + Compute OOF DeLong & McNemar tests (N=8,887)
    print("\n=== Running 5-Fold CV SOTA OOF Predictions (including NeuroPred-FRL) & DeLong/McNemar Tests ===")
    oof_y = []
    oof_probs = {
        "NeuroEGNN": [],
        "PredNeuroP": [],
        "NeuroPred-FRL": [],
        "NeuroPred-CLQ": [],
        "UniDL4BioPep": [],
        "NeuroPred-PLM": [],
        "ESM-2 (650M) + LR": [],
    }
    frl_fold_metrics = []
    for fold_idx in range(5):
        f_num = fold_idx + 1
        tr_f_csv = os.path.join(CODE_DIR, f"training_v2_fold{f_num}.csv")
        te_f_csv = os.path.join(CODE_DIR, f"testing_v2_fold{f_num}.csv")
        tr_f_npz = os.path.join(CODE_DIR, "cache", "esm", f"training_v2_fold{f_num}_t33_650M_L50.npz")
        te_f_npz = os.path.join(CODE_DIR, "cache", "esm", f"testing_v2_fold{f_num}_t33_650M_L50.npz")

        tr_f = extract_plm_multiscale_conv_features(tr_f_csv, tr_f_npz)
        te_f = extract_plm_multiscale_conv_features(te_f_csv, te_f_npz)
        yf_tr, yf_te = tr_f["labels"], te_f["labels"]
        sf_tr, sf_te = tr_f["raw_seqs"], te_f["raw_seqs"]
        nm1_f = np.array([not str(s).strip().startswith("M") for s in sf_te])

        Xf_tr_ms = extract_multiscale_seq_features(sf_tr)
        Xf_te_ms = extract_multiscale_seq_features(sf_te)

        pnp = StackingClassifier(
            estimators=[
                ("gbdt", HistGradientBoostingClassifier(max_iter=200, random_state=42 + fold_idx)),
                ("rf", RandomForestClassifier(n_estimators=300, random_state=42 + fold_idx, n_jobs=-1)),
                ("et", ExtraTreesClassifier(n_estimators=300, random_state=42 + fold_idx, n_jobs=-1)),
            ],
            final_estimator=LogisticRegression(C=1.0, max_iter=500, random_state=42),
            cv=3, n_jobs=-1
        )
        pnp.fit(Xf_tr_ms, yf_tr)
        prob_pnp = pnp.predict_proba(Xf_te_ms)[:, 1]

        frl_mlp = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.5, max_iter=300, early_stopping=True, random_state=42 + fold_idx)
        )
        frl_mlp.fit(Xf_tr_ms, yf_tr)
        prob_frl = 0.5 * frl_mlp.predict_proba(Xf_te_ms)[:, 1] + 0.5 * prob_pnp

        clq = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu", alpha=0.2,
                          batch_size=128, max_iter=200, early_stopping=True, random_state=42 + fold_idx)
        )
        clq.fit(Xf_tr_ms, yf_tr)
        prob_clq = clq.predict_proba(Xf_te_ms)[:, 1]

        Xf_tr_uni = np.concatenate([tr_f["mean"], tr_f["std"]], axis=1)
        Xf_te_uni = np.concatenate([te_f["mean"], te_f["std"]], axis=1)
        uni = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 64), activation="relu", alpha=1.0,
                          batch_size=128, max_iter=200, early_stopping=True, random_state=42 + fold_idx)
        )
        uni.fit(Xf_tr_uni, yf_tr)
        prob_uni = uni.predict_proba(Xf_te_uni)[:, 1]

        Xf_tr_plm = np.concatenate([tr_f["mean"], tr_f["max"], 0.5 * (tr_f["nterm"] + tr_f["cterm"])], axis=1)
        Xf_te_plm = np.concatenate([te_f["mean"], te_f["max"], 0.5 * (te_f["nterm"] + te_f["cterm"])], axis=1)
        npplm = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", alpha=1.5,
                          batch_size=128, max_iter=200, early_stopping=True, random_state=42 + fold_idx)
        )
        npplm.fit(Xf_tr_plm, yf_tr)
        prob_npplm = npplm.predict_proba(Xf_te_plm)[:, 1]

        Xf_tr_lr = np.concatenate([tr_f["mean"], tr_f["max"]], axis=1)
        Xf_te_lr = np.concatenate([te_f["mean"], te_f["max"]], axis=1)
        esmlr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500, random_state=42))
        esmlr.fit(Xf_tr_lr, yf_tr)
        prob_esmlr = esmlr.predict_proba(Xf_te_lr)[:, 1]

        egnn_df = pd.read_csv(os.path.join(
            CODE_DIR, "results", "5fold", "results_5fold", "test", f"fold_{fold_idx}", "test_predictions.csv"
        ))
        prob_egnn = egnn_df["Prediction_Probability"].values

        oof_y.append(yf_te)
        oof_probs["NeuroEGNN"].append(prob_egnn)
        oof_probs["PredNeuroP"].append(prob_pnp)
        oof_probs["NeuroPred-FRL"].append(prob_frl)
        oof_probs["NeuroPred-CLQ"].append(prob_clq)
        oof_probs["UniDL4BioPep"].append(prob_uni)
        oof_probs["NeuroPred-PLM"].append(prob_npplm)
        oof_probs["ESM-2 (650M) + LR"].append(prob_esmlr)

        m_frl = compute_metrics(yf_te, prob_frl)
        m_frl_nm1 = compute_metrics(yf_te[nm1_f], prob_frl[nm1_f])
        frl_fold_metrics.append({"full": m_frl, "nm1": m_frl_nm1})

    egnn_cv_df = pd.read_csv(os.path.join(CODE_DIR, "results", "jcim_supplements", "cv5_sota_and_non_met1_per_fold.csv"))
    egnn_mccs = egnn_cv_df[egnn_cv_df["Method"] == "NeuroEGNN"]["Full_MCC"].values
    frl_mccs = np.array([x["full"]["MCC"] for x in frl_fold_metrics])
    _, p_frl_mcc = stats.ttest_rel(egnn_mccs, frl_mccs)
    frl_summary = {
        k: [float(np.mean([x["full"][k] for x in frl_fold_metrics])),
            float(np.std([x["full"][k] for x in frl_fold_metrics], ddof=1))]
        for k in ["Acc", "Prec", "Sn", "Sp", "BAcc", "MCC", "AUC", "AUPRC", "F1"]
    }
    frl_summary["p_val_MCC"] = float(p_frl_mcc)
    frl_summary["nm1_MCC"] = [
        float(np.mean([x["nm1"]["MCC"] for x in frl_fold_metrics])),
        float(np.std([x["nm1"]["MCC"] for x in frl_fold_metrics], ddof=1))
    ]
    out["cv5_frl"] = frl_summary
    print("  NeuroPred-FRL CV5 Summary:", frl_summary)

    oof_y_all = np.concatenate(oof_y)
    oof_egnn_all = np.concatenate(oof_probs["NeuroEGNN"])
    oof_stats = {}
    print("\n=== Pooled Out-of-Fold (N=8,887) DeLong AUC & McNemar Tests vs NeuroEGNN ===")
    for m_name, plist in oof_probs.items():
        if m_name == "NeuroEGNN":
            continue
        p_all = np.concatenate(plist)
        p_delong, d_auc = delong_p_value(oof_y_all, oof_egnn_all, p_all)
        p_mcnemar, b, c = mcnemar_p_value(oof_y_all, oof_egnn_all >= 0.5, p_all >= 0.5)
        oof_stats[m_name] = {"delong_p": p_delong, "delta_auc": d_auc, "mcnemar_p": p_mcnemar, "b": b, "c": c}
        print(f"  vs {m_name:20s}: ΔAUC={d_auc:+.4f} (DeLong p={p_delong:.3e}) | McNemar b={b}, c={c} (p={p_mcnemar:.3e})")
    out["oof_significance"] = oof_stats

    # Holdout DeLong & McNemar tests (3-seed mean probabilities on N=888)
    print("\n=== Hold-Out (N=888) DeLong & McNemar Tests vs NeuroEGNN (3-seed ensemble & seed_500) ===")
    egnn_ho_ens = np.mean([seed_outs[s]["probs"] for s in (500, 501, 502)], axis=0)
    ho_stats = {}
    for var in ["no_coord", "no_attn", "no_3d", "no_esm"]:
        v_probs = np.mean([
            pd.read_csv(os.path.join(CODE_DIR, "results", "ablation", var, f"seed_{s}", "test", "test_predictions.csv"))["Prediction_Probability"].values
            for s in (500, 501, 502)
        ], axis=0)
        p_delong, d_auc = delong_p_value(y_te, egnn_ho_ens, v_probs)
        p_mcnemar, b, c = mcnemar_p_value(y_te, egnn_ho_ens >= 0.5, v_probs >= 0.5)
        ho_stats[var] = {"delong_p": p_delong, "delta_auc": d_auc, "mcnemar_p": p_mcnemar, "b": b, "c": c}
        print(f"  vs {var:12s}: ΔAUC={d_auc:+.4f} (DeLong p={p_delong:.3e}) | McNemar b={b}, c={c} (p={p_mcnemar:.3e})")
    out["holdout_ablation_significance"] = ho_stats

    # Save OOF predictions and clean diagnostics for figure generation
    np.savez_compressed(
        os.path.join(CODE_DIR, "results", "figure_data", "clean_oof_and_holdout.npz"),
        oof_y=oof_y_all,
        **{f"oof_{k.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')}": np.concatenate(v) for k, v in oof_probs.items()},
        ho_y=y_te,
        ho_egnn_s500=seed_outs[500]["probs"],
        ho_egnn_ens=egnn_ho_ens,
        s2g_s500=seed_outs[500]["attn_s2g"],
        g2s_s500=seed_outs[500]["attn_g2s"],
        s2g_ens=np.mean([seed_outs[s]["attn_s2g"] for s in (500, 501, 502)], axis=0),
        g2s_ens=np.mean([seed_outs[s]["attn_g2s"] for s in (500, 501, 502)], axis=0),
    )
    with open(os.path.join(CODE_DIR, "results", "figure_data", "clean_diagnostics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved clean_diagnostics.json and clean_oof_and_holdout.npz successfully.")


if __name__ == "__main__":
    main()
