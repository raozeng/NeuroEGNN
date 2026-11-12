import os, sys, json, numpy as np, pandas as pd
from scipy import stats
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from run_debiased_and_relaxation import AAS, load_ckpt, load_batch_tensors, forward_neuroegnn_numpy

s2g_stats = {"local_seq": [], "med_spatial": [], "long_spatial": [], "distant": []}
g2s_stats = {"local_seq": [], "med_spatial": [], "long_spatial": [], "distant": []}
aa_tp = {a: [] for a in AAS}
aa_tn = {a: [] for a in AAS}
aa_tp_int = {a: [] for a in AAS}
m1_tn = []
tp_seg = {"N": [], "Core": [], "C": []}
fn_m1_fracs = []
rel_pos_tp = [[] for _ in range(20)]
rel_pos_tn = [[] for _ in range(20)]

for f_idx in range(5):
    ck = load_ckpt(os.path.join(CODE_DIR, "results", "5fold", "results_5fold", "train", f"fold_{f_idx}", "best_model.pth"))
    te_csv = os.path.join(CODE_DIR, f"testing_v2_fold{f_idx+1}.csv")
    te_npz = os.path.join(CODE_DIR, "cache", "esm", f"testing_v2_fold{f_idx+1}_t33_650M_L50.npz")
    bd = load_batch_tensors(te_csv, te_npz)
    n_tot = len(bd["labels"])
    for st in range(0, n_tot, 256):
        ed = min(n_tot, st + 256)
        sub_bd = {k: (v[st:ed] if isinstance(v, (np.ndarray, list)) else v) for k, v in bd.items()}
        out = forward_neuroegnn_numpy(ck["state_dict"], ck["config"], sub_bd, return_details=True)
        probs = out["probs"]
        preds = (probs >= 0.5).astype(int)
        for i in range(ed - st):
            L = int(sub_bd["lengths"][i])
            cs = sub_bd["cleaned_seqs"][i]
            y, pr = int(sub_bd["labels"][i]), int(preds[i])
            s2g = out["attn_s2g"][i, :L, :L]
            g2s = out["attn_g2s"][i, :L, :L]
            res = 0.5 * (s2g.mean(axis=0) + g2s.mean(axis=0))
            norm = res / (np.mean(res) + 1e-8)
            if y == 1 and pr == 1:
                for r_i, ch in enumerate(cs):
                    if ch in aa_tp:
                        aa_tp[ch].append(float(norm[r_i]))
                        if r_i >= 1: aa_tp_int[ch].append(float(norm[r_i]))
                    b_idx = min(19, int(np.floor(r_i / max(L, 1) * 20)))
                    rel_pos_tp[b_idx].append(float(norm[r_i]))
                if L >= 9:
                    k = min(3, L // 3)
                    tp_seg["N"].append(float(np.mean(norm[:k])))
                    tp_seg["Core"].append(float(np.mean(norm[k:-k])))
                    tp_seg["C"].append(float(np.mean(norm[-k:])))
            elif y == 0 and pr == 0:
                for r_i, ch in enumerate(cs):
                    if ch in aa_tn: aa_tn[ch].append(float(norm[r_i]))
                    if r_i == 0 and ch == "M": m1_tn.append(float(norm[0]))
                    b_idx = min(19, int(np.floor(r_i / max(L, 1) * 20)))
                    rel_pos_tn[b_idx].append(float(norm[r_i]))
            elif y == 1 and pr == 0 and cs.startswith("M"):
                fn_m1_fracs.append(float(res[0] / (np.sum(res) + 1e-8)))
            if L >= 6:
                coords = sub_bd["coords"][i, :L]
                dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
                ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
                sep = np.abs(ii - jj)
                for key, mk in [("local_seq", (sep>=1)&(sep<=2)), ("med_spatial", (sep>=3)&(sep<=5)&(dist<10.0)),
                                ("long_spatial", (sep>5)&(dist<10.0)), ("distant", (sep>2)&(dist>=10.0))]:
                    if np.any(mk):
                        s2g_stats[key].append(float(np.mean(s2g[mk])*L))
                        g2s_stats[key].append(float(np.mean(g2s[mk])*L))

print("=== 5-Fold Out-Of-Fold (N=8,887, 100% Zero-Leakage) Interpretability ===")
for k in ["local_seq", "med_spatial", "long_spatial", "distant"]:
    print(f"  {k:14s}: s2g={np.mean(s2g_stats[k]):.4f}±{np.std(s2g_stats[k]):.4f} | g2s={np.mean(g2s_stats[k]):.4f}±{np.std(g2s_stats[k]):.4f}")
lr_pct = (np.mean(s2g_stats['long_spatial']) / np.mean(s2g_stats['local_seq']) - 1) * 100
c_pct = (np.mean(tp_seg['C']) / np.mean(tp_seg['Core']) - 1) * 100
print(f"  Long vs Local s2g: +{lr_pct:.2f}%")
print(f"  TP Segments (n={len(tp_seg['N'])}): N={np.mean(tp_seg['N']):.4f}±{np.std(tp_seg['N']):.4f}, Core={np.mean(tp_seg['Core']):.4f}±{np.std(tp_seg['Core']):.4f}, C={np.mean(tp_seg['C']):.4f}±{np.std(tp_seg['C']):.4f} (C/Core=+{c_pct:.2f}%)")
rows = [(a, np.mean(aa_tp[a]), np.mean(aa_tn[a]), np.mean(aa_tp_int[a]), len(aa_tp[a]), len(aa_tn[a])) for a in AAS]
rows.sort(key=lambda x: x[1], reverse=True)
for a, tp_m, tn_m, tp_i, ntp, ntn in rows:
    print(f"    {a}: TP={tp_m:.4f} (int={tp_i:.4f}, n={ntp}) | TN={tn_m:.4f} (n={ntn}) | TP/TN={tp_m/tn_m:.2f}x")
print(f"  TN Met1 pos1 mean={np.mean(m1_tn):.4f} (n={len(m1_tn)}) | FN M1 share={np.mean(fn_m1_fracs)*100:.2f}% (n={len(fn_m1_fracs)})")

p_full = np.mean([pd.read_csv(os.path.join(CODE_DIR, "results", "ablation", "full", f"seed_{s}", "test", "test_predictions.csv"))["Prediction_Probability"].values for s in (500,501,502)], axis=0)
p_no3d = np.mean([pd.read_csv(os.path.join(CODE_DIR, "results", "ablation", "no_3d", f"seed_{s}", "test", "test_predictions.csv"))["Prediction_Probability"].values for s in (500,501,502)], axis=0)
y_te = pd.read_csv(os.path.join(CODE_DIR, "testing_v2.csv"))["Label"].values
pos_m = (y_te == 1)
c_f = (p_full[pos_m] >= 0.5)
c_n = (p_no3d[pos_m] >= 0.5)
b = int(np.sum(c_f & (~c_n)))
c = int(np.sum((~c_f) & c_n))
print(f"\nPositive Sensitivity McNemar (full vs no_3d, n=442): b={b}, c={c}, p={stats.binomtest(min(b,c), b+c, 0.5).pvalue:.3e}")

np.savez_compressed(
    os.path.join(CODE_DIR, "results", "figure_data", "oof_interp_8887.npz"),
    s2g_local=np.array(s2g_stats["local_seq"]),
    s2g_med=np.array(s2g_stats["med_spatial"]),
    s2g_long=np.array(s2g_stats["long_spatial"]),
    s2g_dist=np.array(s2g_stats["distant"]),
    g2s_local=np.array(g2s_stats["local_seq"]),
    g2s_med=np.array(g2s_stats["med_spatial"]),
    g2s_long=np.array(g2s_stats["long_spatial"]),
    g2s_dist=np.array(g2s_stats["distant"]),
    aa_names=np.array(list(AAS)),
    aa_tp_means=np.array([np.mean(aa_tp[a]) for a in AAS]),
    aa_tn_means=np.array([np.mean(aa_tn[a]) for a in AAS]),
    aa_tp_int_means=np.array([np.mean(aa_tp_int[a]) for a in AAS]),
    rel_tp_means=np.array([np.mean(x) for x in rel_pos_tp]),
    rel_tp_sems=np.array([np.std(x)/np.sqrt(len(x)) for x in rel_pos_tp]),
    rel_tn_means=np.array([np.mean(x) for x in rel_pos_tn]),
    rel_tn_sems=np.array([np.std(x)/np.sqrt(len(x)) for x in rel_pos_tn]),
)
