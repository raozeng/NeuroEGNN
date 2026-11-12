# -*- coding: utf-8 -*-
"""
Regenerate all 7 publication-grade figures (Figures 2-7 + TOC Graphic)
with 100% zero-leakage statistics, unified uppercase (A)-(D) panel labels,
and exact alignment with main.tex tables and captions.
Note: fig0_architecture.pdf/.png (Figure 1) is intentionally preserved untouched.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import fitz
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_PAPER_FIG = os.path.join(os.path.dirname(CODE_DIR), "paper", "figures")
FIG_DIR = _PAPER_FIG if os.path.isdir(_PAPER_FIG) else os.path.join(CODE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
DATA_DIR = os.path.join(CODE_DIR, "results", "figure_data")
sys.path.insert(0, CODE_DIR)

from make_split import cluster_sequences
from run_debiased_and_relaxation import (
    AAS, extract_handcrafted_430, extract_multiscale_seq_features,
    extract_plm_multiscale_conv_features, load_ckpt, load_batch_tensors,
    forward_neuroegnn_numpy
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.5,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.6,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})

NAVY = "#1D3557"
CRIMSON = "#D62828"
TEAL = "#2A9D8F"
STEEL = "#457B9D"
AMBER = "#E76F51"
GOLD = "#E9C46A"
PLUM = "#6D597A"
SLATE = "#64748B"

with open(os.path.join(DATA_DIR, "clean_diagnostics.json")) as f:
    DIAG = json.load(f)
OOF_NPZ = np.load(os.path.join(DATA_DIR, "clean_oof_and_holdout.npz"), allow_pickle=True)
OOF_INTERP = np.load(os.path.join(DATA_DIR, "oof_interp_8887.npz"), allow_pickle=True)


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ===========================================================================
# 1. Figure 2: fig1_dataset_analysis.pdf / .png
# ===========================================================================
def make_fig1_dataset():
    print("[1/7] Generating fig1_dataset_analysis ...")
    df_all = pd.read_csv(os.path.join(CODE_DIR, "all_data.csv"))
    seqs = df_all["Seq"].astype(str).str.strip().tolist()
    labels = df_all["Label"].astype(int).values
    lens = np.array([len(s) for s in seqs])

    pos_lens = lens[labels == 1]
    neg_lens = lens[labels == 0]

    # Cluster sizes (from exact cached single-linkage clustering C = 7,050)
    cids = np.load(os.path.join(CODE_DIR, "cache", "clusters_sim0.8_7f3749783ffd.npy"))
    _, counts = np.unique(cids, return_counts=True)

    fig = plt.figure(figsize=(12.6, 8.8))
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.34)

    # (A) Sequence length distribution
    axA = fig.add_subplot(gs[0, 0])
    bins = np.arange(5, 102, 3)
    axA.hist(neg_lens, bins=bins, density=True, alpha=0.35, color=STEEL, edgecolor=STEEL, lw=0.8,
             label=f"Non-Neuropeptide ($n=4,463$, {neg_lens.mean():.1f}±{neg_lens.std():.1f} aa)")
    axA.hist(pos_lens, bins=bins, density=True, alpha=0.38, color=CRIMSON, edgecolor=CRIMSON, lw=0.8,
             label=f"Neuropeptide ($n=4,424$, {pos_lens.mean():.1f}±{pos_lens.std():.1f} aa)")
    xs = np.linspace(6, 99, 300)
    axA.plot(xs, gaussian_kde(neg_lens, bw_method=0.22)(xs), color=NAVY, lw=1.8)
    axA.plot(xs, gaussian_kde(pos_lens, bw_method=0.22)(xs), color=CRIMSON, lw=2.0)
    axA.axvline(50, color="#0F766E", ls="--", lw=1.5)
    axA.text(51.5, 0.066, "$L_{\\max} = 50$ aa\n(91.49% full-length)", color="#0F766E",
             fontsize=7.8, fontweight="bold", va="top")
    axA.axvline(15.5, color="#94A3B8", ls=":", lw=1.0)
    axA.axvline(30.5, color="#94A3B8", ls=":", lw=1.0)
    axA.set_xlabel("Peptide Sequence Length $L$ (aa)")
    axA.set_ylabel("Probability Density")
    axA.set_xlim(4, 101)
    axA.set_ylim(0, 0.092)
    axA.set_title("(A) Sequence Length Distributions Across Benchmark Classes", loc="left", fontweight="bold", pad=8)
    axA.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.4)
    axA.grid(axis="y", ls="--", alpha=0.28)
    despine(axA)

    # (B) Homology cluster size distribution
    axB = fig.add_subplot(gs[0, 1])
    size_bins = [1, 2, 3, 4, 5, 10, 20, 50, 200]
    bin_labels = ["1\n(Singleton)", "2", "3", "4", "5–9", "10–19", "20–49", "50–191"]
    bucket_counts = [
        int(np.sum(counts == 1)),
        int(np.sum(counts == 2)),
        int(np.sum(counts == 3)),
        int(np.sum(counts == 4)),
        int(np.sum((counts >= 5) & (counts <= 9))),
        int(np.sum((counts >= 10) & (counts <= 19))),
        int(np.sum((counts >= 20) & (counts <= 49))),
        int(np.sum(counts >= 50)),
    ]
    x_b = np.arange(len(bin_labels))
    colors_b = [STEEL] + [TEAL] * 7
    bars_b = axB.bar(x_b, bucket_counts, width=0.64, color=colors_b, edgecolor=NAVY, lw=0.7, alpha=0.9)
    axB.set_yscale("log")
    axB.set_ylim(0.7, 16000)
    for rect, cnt in zip(bars_b, bucket_counts):
        axB.text(rect.get_x() + rect.get_width() / 2, cnt * 1.22, f"{cnt:,}",
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1E293B")
    axB.text(0.97, 0.88,
             "Total: $C = 7,050$ clusters ($\\tau = 0.80$)\n"
             "• 6,526 singletons (73.43% of peptides)\n"
             "• 524 multi-member families (2,361 peptides,\n"
             "  26.57% at risk under random splits; max = 191)",
             transform=axB.transAxes, ha="right", va="top", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.9))
    axB.set_xticks(x_b)
    axB.set_xticklabels(bin_labels, fontsize=7.7)
    axB.set_xlabel("Cluster Cardinality $|\\mathcal{C}_c|$ (Sequences per Homology Family)")
    axB.set_ylabel("Number of Clusters (Log Scale)")
    axB.set_title("(B) Single-Linkage Homology Cluster Distribution ($\\tau = 0.80$)", loc="left", fontweight="bold", pad=8)
    axB.grid(axis="y", ls="--", alpha=0.28)
    despine(axB)

    # (C) Global amino acid log2 enrichment
    axC = fig.add_subplot(gs[1, 0])
    pos_str = "".join([s for s, y in zip(seqs, labels) if y == 1])
    neg_str = "".join([s for s, y in zip(seqs, labels) if y == 0])
    log2_fc = []
    for a in AAS:
        fp = pos_str.count(a) / len(pos_str)
        fn = neg_str.count(a) / len(neg_str)
        log2_fc.append((a, np.log2(fp / fn)))
    log2_fc.sort(key=lambda x: x[1])
    aa_sorted = [x[0] for x in log2_fc]
    fc_vals = np.array([x[1] for x in log2_fc])
    c_bars = [CRIMSON if v >= 0 else STEEL for v in fc_vals]
    y_c = np.arange(len(aa_sorted))
    axC.barh(y_c, fc_vals, height=0.68, color=c_bars, edgecolor="#1E293B", lw=0.5, alpha=0.88)
    axC.axvline(0, color="#0F172A", lw=0.9)
    axC.set_yticks(y_c)
    axC.set_yticklabels(aa_sorted, fontweight="bold", fontsize=7.8)
    axC.set_xlim(-1.18, 0.72)
    axC.set_xlabel("Global Compositional Log$_2$ Enrichment $\\log_2(f_{\\mathrm{Pos}} / f_{\\mathrm{Neg}})$")
    axC.set_title("(C) Residue-Level Compositional Enrichment Between Classes", loc="left", fontweight="bold", pad=8)
    axC.text(-1.12, 15.2, "Enriched in Negatives\n(Ile, Met, Lys, Val)", fontsize=7.4, color=NAVY, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1", alpha=0.92))
    axC.text(0.06, 1.4, "Enriched in Neuropeptides\n(Pro, Gly, Asp, Glu, Trp, Phe)", fontsize=7.4, color=CRIMSON, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#FECACA", alpha=0.92))
    axC.grid(axis="x", ls="--", alpha=0.28)
    despine(axC)

    # (D) Positional Met enrichment across positions 1..15
    axD = fig.add_subplot(gs[1, 1])
    positions = np.arange(1, 16)
    pos_m_freq, neg_m_freq = [], []
    for p_idx in range(15):
        p_sub = [s[p_idx] == "M" for s, y in zip(seqs, labels) if y == 1 and len(s) > p_idx]
        n_sub = [s[p_idx] == "M" for s, y in zip(seqs, labels) if y == 0 and len(s) > p_idx]
        pos_m_freq.append(np.mean(p_sub) * 100.0)
        neg_m_freq.append(np.mean(n_sub) * 100.0)
    w_d = 0.38
    axD.bar(positions - w_d / 2, neg_m_freq, width=w_d, color=STEEL, edgecolor=NAVY, lw=0.6,
            label="Negative Controls")
    axD.bar(positions + w_d / 2, pos_m_freq, width=w_d, color=CRIMSON, edgecolor="#7F1D1D", lw=0.6,
            label="Mature Neuropeptides")
    axD.annotate("65.47% Neg vs. 1.99% Pos\n(32.9× N-terminal $\\mathrm{Met}_1$ shortcut!)",
                 xy=(1 - w_d / 2, 65.5), xytext=(3.3, 55.0),
                 fontsize=8.0, fontweight="bold", color=NAVY,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF2F2", edgecolor=CRIMSON, lw=0.9),
                 arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.2))
    axD.set_xticks(positions)
    axD.set_ylim(0, 74)
    axD.set_xlabel("Residue Position from N-Terminus ($i = 1, \\dots, 15$)")
    axD.set_ylabel("Methionine Frequency at Position $i$ (%)")
    axD.set_title("(D) Positional N-Terminal Initiator $\\mathrm{Met}_1$ Cleavage Bias", loc="left", fontweight="bold", pad=8)
    axD.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92)
    axD.grid(axis="y", ls="--", alpha=0.28)
    despine(axD)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig1_dataset_analysis.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ===========================================================================
# 2. Figure 3: fig2_performance_comparison.pdf / .png
# ===========================================================================
def make_fig2_performance():
    print("[2/7] Generating fig2_performance_comparison ...")
    hpc = np.load(os.path.join(CODE_DIR, "results", "baselines", "holdout_predictions_curves.npz"), allow_pickle=True)
    y_te = OOF_NPZ["ho_y"]
    p_egnn_ens = OOF_NPZ["ho_egnn_ens"]

    # Compute clean holdout SOTA curves on training_v2 -> testing_v2
    tr_csv = os.path.join(CODE_DIR, "training_v2.csv")
    te_csv = os.path.join(CODE_DIR, "testing_v2.csv")
    tr_npz = os.path.join(CODE_DIR, "cache", "esm", "training_v2_t33_650M_L50.npz")
    te_npz = os.path.join(CODE_DIR, "cache", "esm", "testing_v2_t33_650M_L50.npz")
    tr_plm = extract_plm_multiscale_conv_features(tr_csv, tr_npz)
    te_plm = extract_plm_multiscale_conv_features(te_csv, te_npz)
    y_tr = tr_plm["labels"]
    X_tr_ms = extract_multiscale_seq_features(tr_plm["raw_seqs"])
    X_te_ms = extract_multiscale_seq_features(te_plm["raw_seqs"])

    # PredNeuroP
    pnp = StackingClassifier(
        estimators=[
            ("gbdt", HistGradientBoostingClassifier(max_iter=200, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            ("et", ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            ("svm", make_pipeline(StandardScaler(), SVC(C=2.0, kernel="rbf", probability=True, random_state=42))),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=500, random_state=42),
        cv=3, n_jobs=-1
    ).fit(X_tr_ms, y_tr)
    prob_pnp = pnp.predict_proba(X_te_ms)[:, 1]

    # NeuroPred-FRL
    views = [slice(0, 20), slice(20, 420), slice(420, 430), slice(430, 470), slice(470, 520)]
    meta_tr, meta_te = [], []
    for sl in views:
        for clf in [
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500, random_state=42)),
        ]:
            clf.fit(X_tr_ms[:, sl], y_tr)
            meta_tr.append(clf.predict_proba(X_tr_ms[:, sl])[:, 1])
            meta_te.append(clf.predict_proba(X_te_ms[:, sl])[:, 1])
    frl_meta = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(np.column_stack(meta_tr), y_tr)
    prob_frl = frl_meta.predict_proba(np.column_stack(meta_te))[:, 1]

    # 3-seed average probabilities for neural SOTA baselines (matching Table 4 Mean ± SD)
    X_tr_uni = np.concatenate([tr_plm["mean"], tr_plm["max"], tr_plm["std"]], axis=1)
    X_te_uni = np.concatenate([te_plm["mean"], te_plm["max"], te_plm["std"]], axis=1)
    X_tr_npplm = np.concatenate([tr_plm["mean"], tr_plm["max"], 0.5 * (tr_plm["nterm"] + tr_plm["cterm"])], axis=1)
    X_te_npplm = np.concatenate([te_plm["mean"], te_plm["max"], 0.5 * (te_plm["nterm"] + te_plm["cterm"])], axis=1)

    probs_clq, probs_uni, probs_npplm = [], [], []
    for sd in (500, 501, 502):
        c_clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128, 64), alpha=0.5, batch_size=128, max_iter=200, early_stopping=True, random_state=sd)).fit(X_tr_ms, y_tr)
        probs_clq.append(c_clf.predict_proba(X_te_ms)[:, 1])
        u_clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1.0, batch_size=128, max_iter=200, early_stopping=True, random_state=sd)).fit(X_tr_uni, y_tr)
        probs_uni.append(u_clf.predict_proba(X_te_uni)[:, 1])
        n_clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1.5, batch_size=128, max_iter=200, early_stopping=True, random_state=sd)).fit(X_tr_npplm, y_tr)
        probs_npplm.append(n_clf.predict_proba(X_te_npplm)[:, 1])

    prob_clq = np.mean(probs_clq, axis=0)
    prob_uni = np.mean(probs_uni, axis=0)
    prob_npplm = np.mean(probs_npplm, axis=0)
    prob_esmlr = hpc["ESM-2_650M___LR"]
    prob_gbdt = hpc["AAC_DPC_PCP___GBDT"]

    # Note: report exact 3-seed mean AUC/AUPRC from Table 4 in legend for 100% consistency
    curve_specs = [
        ("NeuroEGNN (Ours)", p_egnn_ens, CRIMSON, "-", 2.4, 0.9765, 0.9761),
        ("NeuroPred-PLM (2023)", prob_npplm, NAVY, "-", 1.7, 0.9637, 0.9627),
        ("UniDL4BioPep (2023)", prob_uni, TEAL, "--", 1.7, 0.9637, 0.9629),
        ("PredNeuroP (2020)", prob_pnp, AMBER, "-.", 1.5, 0.9562, 0.9542),
        ("ESM-2 (650M) + LR", prob_esmlr, STEEL, ":", 1.6, 0.9551, 0.9536),
        ("NeuroPred-FRL (2021)", prob_frl, PLUM, "--", 1.4, 0.9504, 0.9477),
        ("AAC+DPC+PCP + GBDT", prob_gbdt, "#6B7280", "-.", 1.3, 0.9368, 0.9301),
        ("NeuroPred-CLQ (2022)", prob_clq, "#94A3B8", ":", 1.3, 0.9245, 0.9196),
    ]

    fig = plt.figure(figsize=(12.8, 9.2))
    gs = fig.add_gridspec(2, 2, wspace=0.24, hspace=0.36)

    # (A) ROC Curves with non-overlapping zoomed inset (upper-right middle) and legend (lower-right)
    axA = fig.add_subplot(gs[0, 0])
    axA.plot([0, 1], [0, 1], color="#CBD5E1", ls="--", lw=1.0)
    ax_ins = axA.inset_axes([0.45, 0.50, 0.50, 0.39])
    for name, pr, col, ls, lw, auc_lbl, _ in curve_specs:
        fpr, tpr, _ = roc_curve(y_te, pr)
        axA.plot(fpr, tpr, color=col, ls=ls, lw=lw, label=f"{name} ({auc_lbl:.4f})")
        ax_ins.plot(fpr, tpr, color=col, ls=ls, lw=lw)
    ax_ins.set_xlim(0.0, 0.20)
    ax_ins.set_ylim(0.80, 1.00)
    ax_ins.tick_params(labelsize=6.2)
    ax_ins.grid(True, ls=":", alpha=0.35)
    ax_ins.set_title("High-Sp Zoom (FPR $\\leq 0.20$)", fontsize=6.5, fontweight="bold", pad=3)
    axA.set_xlim(-0.01, 1.01)
    axA.set_ylim(-0.01, 1.02)
    axA.set_xlabel("False Positive Rate ($1 - \\mathrm{Specificity}$)")
    axA.set_ylabel("True Positive Rate (Sensitivity)")
    axA.set_title("(A) Hold-Out ROC Curves ($N = 888$)", loc="left", fontweight="bold", pad=8)
    axA.legend(loc="lower right", fontsize=6.4, labelspacing=0.28, frameon=True, facecolor="white", framealpha=0.95)
    axA.grid(True, ls="--", alpha=0.26)
    despine(axA)

    # (B) PR Curves
    axB = fig.add_subplot(gs[0, 1])
    for name, pr, col, ls, lw, _, auprc_lbl in curve_specs:
        prec, rec, _ = precision_recall_curve(y_te, pr)
        axB.plot(rec, prec, color=col, ls=ls, lw=lw, label=f"{name} ({auprc_lbl:.4f})")
    axB.set_xlim(0.0, 1.01)
    axB.set_ylim(0.52, 1.015)
    axB.set_xlabel("Recall (Sensitivity)")
    axB.set_ylabel("Precision (Positive Predictive Value)")
    axB.set_title("(B) Hold-Out Precision–Recall Curves ($N = 888$)", loc="left", fontweight="bold", pad=8)
    axB.legend(loc="lower left", fontsize=6.8, frameon=True, facecolor="white", framealpha=0.94)
    axB.grid(True, ls="--", alpha=0.26)
    despine(axB)

    # (C) 5-Fold CV Multi-Metric Comparison
    axC = fig.add_subplot(gs[1, 0])
    df_cv_base = pd.read_csv(os.path.join(CODE_DIR, "results", "baselines", "cv5_baselines_summary.csv")).set_index("Method")
    df_cv_sota = pd.read_csv(os.path.join(CODE_DIR, "results", "debiased_and_relaxation", "cv5_sota_and_non_met1_summary.csv")).set_index("Method")

    cv_items = [
        ("GBDT\n(1D)", df_cv_base.loc["AAC+DPC+PCP + GBDT"], "base"),
        ("3D-Geo\n+GBDT", df_cv_base.loc["3D-Geo+PCP + GBDT"], "base"),
        ("NP-CLQ", df_cv_sota.loc["NeuroPred-CLQ (Bioinf 2022)"], "sota"),
        ("NP-FRL", df_cv_sota.loc["NeuroPred-FRL (BiB 2021)"], "sota"),
        ("Pred-\nNeuroP", df_cv_sota.loc["PredNeuroP (BiB 2020)"], "sota"),
        ("ESM-2\n+LR", df_cv_base.loc["ESM-2 (650M) + LR"], "base"),
        ("NP-PLM", df_cv_sota.loc["NeuroPred-PLM (BiB 2024)"], "sota"),
        ("UniDL4\nBioPep", df_cv_sota.loc["UniDL4BioPep (BiB 2023)"], "sota"),
        ("NeuroEGNN\n(Ours)", df_cv_sota.loc["NeuroEGNN"], "sota"),
    ]
    x_c = np.arange(len(cv_items))
    w_c = 0.24
    acc_v = [r["Accuracy_mean"] if t == "base" else r["Full_Acc_mean"] for _, r, t in cv_items]
    sn_v = [r["Sensitivity_mean"] if t == "base" else r["Full_Sn_mean"] for _, r, t in cv_items]
    auc_v = [r["AUC_mean"] if t == "base" else r["Full_AUC_mean"] for _, r, t in cv_items]
    acc_e = [r["Accuracy_std"] if t == "base" else r["Full_Acc_std"] for _, r, t in cv_items]
    sn_e = [r["Sensitivity_std"] if t == "base" else r["Full_Sn_std"] for _, r, t in cv_items]
    auc_e = [r["AUC_std"] if t == "base" else r["Full_AUC_std"] for _, r, t in cv_items]

    axC.bar(x_c - w_c, np.array(acc_v) * 100, w_c, yerr=np.array(acc_e) * 100, capsize=2.0,
            color=STEEL, edgecolor=NAVY, lw=0.6, label="Accuracy (%)")
    axC.bar(x_c, np.array(sn_v) * 100, w_c, yerr=np.array(sn_e) * 100, capsize=2.0,
            color=CRIMSON, edgecolor="#7F1D1D", lw=0.6, label="Sensitivity (%)")
    axC.bar(x_c + w_c, np.array(auc_v) * 100, w_c, yerr=np.array(auc_e) * 100, capsize=2.0,
            color=TEAL, edgecolor="#115E59", lw=0.6, label="AUC-ROC (×100)")
    axC.set_ylim(78, 101.5)
    axC.set_xticks(x_c)
    axC.set_xticklabels([lbl for lbl, _, _ in cv_items], fontsize=7.2)
    axC.set_ylabel("5-Fold Cross-Validation Score (%)")
    axC.set_title("(C) 5-Fold Homology-Isolated Cross-Validation ($N = 8,887$)", loc="left", fontweight="bold", pad=18)
    axC.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False, fontsize=7.5)
    axC.grid(axis="y", ls="--", alpha=0.28)
    despine(axC)

    # (D) Fold-wise MCC box/strip plot + OOF DeLong / McNemar significance
    axD = fig.add_subplot(gs[1, 1])
    df_pf = pd.read_csv(os.path.join(CODE_DIR, "results", "debiased_and_relaxation", "cv5_sota_and_non_met1_per_fold.csv"))
    # Compute per-fold MCC for NeuroPred-FRL from clean_oof_and_holdout.npz
    oof_frl = OOF_NPZ["oof_NeuroPred-FRL"]
    oof_y = OOF_NPZ["oof_y"]
    fold_sizes = [1778, 1778, 1778, 1777, 1776]
    frl_fold_mcc = []
    idx0 = 0
    from sklearn.metrics import matthews_corrcoef
    for fs in fold_sizes:
        yf = oof_y[idx0:idx0 + fs]
        pf = (oof_frl[idx0:idx0 + fs] >= 0.5).astype(int)
        frl_fold_mcc.append(matthews_corrcoef(yf, pf))
        idx0 += fs

    d_methods = [
        ("NeuroPred\n-CLQ", df_pf[df_pf["Method"] == "NeuroPred-CLQ (Bioinf 2022)"]["Full_MCC"].values, "#94A3B8"),
        ("NeuroPred\n-FRL", np.array(frl_fold_mcc), PLUM),
        ("PredNeuroP", df_pf[df_pf["Method"] == "PredNeuroP (BiB 2020)"]["Full_MCC"].values, AMBER),
        ("ESM-2\n+LR", df_pf[df_pf["Method"] == "ESM-2 (650M) + LR"]["Full_MCC"].values, STEEL),
        ("NeuroPred\n-PLM", df_pf[df_pf["Method"] == "NeuroPred-PLM (BiB 2024)"]["Full_MCC"].values, NAVY),
        ("UniDL4\nBioPep", df_pf[df_pf["Method"] == "UniDL4BioPep (BiB 2023)"]["Full_MCC"].values, TEAL),
        ("NeuroEGNN\n(Ours)", df_pf[df_pf["Method"] == "NeuroEGNN"]["Full_MCC"].values, CRIMSON),
    ]
    pos_d = np.arange(len(d_methods))
    bp = axD.boxplot([vals for _, vals, _ in d_methods], positions=pos_d, widths=0.46,
                     patch_artist=True, showfliers=False)
    for patch, (_, _, col) in zip(bp["boxes"], d_methods):
        patch.set_facecolor(col)
        patch.set_alpha(0.32)
        patch.set_edgecolor(col)
        patch.set_linewidth(1.2)
    for med in bp["medians"]:
        med.set_color("#0F172A")
        med.set_linewidth(1.5)
    rng = np.random.RandomState(42)
    for i, (_, vals, col) in enumerate(d_methods):
        jit = rng.uniform(-0.11, 0.11, size=len(vals))
        axD.scatter(np.full_like(vals, i, dtype=float) + jit, vals, color=col, edgecolor="white", s=34, lw=0.6, zorder=4)

    # Significance brackets (OOF DeLong & McNemar)
    axD.plot([5, 5, 6, 6], [0.871, 0.878, 0.878, 0.871], color="#0F172A", lw=1.0)
    axD.text(5.5, 0.880, "McNemar $p=1.8\\times 10^{-5}$\nDeLong $p=4.1\\times 10^{-10}$",
             ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=CRIMSON)
    axD.plot([4, 4, 6, 6], [0.845, 0.910, 0.910, 0.903], color="#334155", lw=0.9)
    axD.text(5.0, 0.912, "vs. NeuroPred-PLM: McNemar $p=9.2\\times 10^{-13}$, DeLong $p=1.5\\times 10^{-17}$",
             ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=NAVY)

    axD.set_xticks(pos_d)
    axD.set_xticklabels([m for m, _, _ in d_methods], fontsize=7.3)
    axD.set_ylim(0.70, 0.942)
    axD.set_ylabel("Fold-Wise MCC ($K = 5$ Folds)")
    axD.set_title("(D) Fold-Wise MCC & Out-of-Fold Significance ($N = 8,887$)", loc="left", fontweight="bold", pad=8)
    axD.grid(axis="y", ls="--", alpha=0.28)
    despine(axD)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig2_performance_comparison.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ===========================================================================
# 3. Figure 4: fig3_ablation_and_length.pdf / .png
# ===========================================================================
def make_fig3_ablation():
    print("[3/7] Generating fig3_ablation_and_length ...")
    df_ab = pd.read_csv(os.path.join(CODE_DIR, "results", "debiased_and_relaxation", "holdout_seeds_sota_and_non_met1.csv")).set_index("Method")
    df_len = pd.read_csv(os.path.join(CODE_DIR, "results", "baselines", "ablation_length_stratified.csv"))

    variants = [
        ("full", "NeuroEGNN (full)", "full\n(0.917M)", CRIMSON),
        ("no_coord", "w/o Coord Update (no_coord)", "no_coord\n(0.851M)", AMBER),
        ("no_attn", "w/o Cross-Attn (no_attn)", "no_attn\n(0.735M)", TEAL),
        ("no_3d", "w/o 3D-EGNN (no_3d)", "no_3d\n(0.271M)", NAVY),
        ("no_esm", "w/o 1D ESM-2 (no_esm)", "no_esm\n(0.571M)", SLATE),
    ]

    fig = plt.figure(figsize=(12.8, 9.0))
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.38)

    # (A) Multi-metric bar chart with legend cleanly above bars
    axA = fig.add_subplot(gs[0, 0])
    x_a = np.arange(len(variants))
    w_a = 0.18
    metrics_a = [
        ("Full_Acc", "Acc (%)", STEEL, 100.0),
        ("Full_Sn", "Sn (%)", CRIMSON, 100.0),
        ("Full_Sp", "Sp (%)", "#6D597A", 100.0),
        ("Full_AUC", "AUC (×100)", TEAL, 100.0),
    ]
    for m_i, (col, lbl, c_bar, scale) in enumerate(metrics_a):
        vals = [df_ab.loc[rk, f"{col}_mean"] * scale for _, rk, _, _ in variants]
        errs = [df_ab.loc[rk, f"{col}_std"] * scale for _, rk, _, _ in variants]
        offset = (m_i - 1.5) * w_a
        axA.bar(x_a + offset, vals, w_a, yerr=errs, capsize=2.0, color=c_bar, edgecolor="#1E293B", lw=0.55, label=lbl)
    axA.set_ylim(76, 100.8)
    axA.set_xticks(x_a)
    axA.set_xticklabels([xl for _, _, xl, _ in variants], fontsize=7.8)
    axA.set_ylabel("Hold-Out Score (%, Mean ± SD)")
    axA.set_title("(A) Architectural Ablation Across 3 Random Seeds", loc="left", fontweight="bold", pad=20)
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=4, frameon=False, fontsize=7.5)
    axA.grid(axis="y", ls="--", alpha=0.28)
    despine(axA)

    # (B) Delta MCC and Delta Sn (percentage points pp)
    axB = fig.add_subplot(gs[0, 1])
    full_mcc = df_ab.loc["NeuroEGNN (full)", "Full_MCC_mean"]
    full_sn = df_ab.loc["NeuroEGNN (full)", "Full_Sn_mean"] * 100.0
    ab_only = variants[1:]
    y_b = np.arange(len(ab_only))[::-1]
    dmcc = [(df_ab.loc[rk, "Full_MCC_mean"] - full_mcc) * 100.0 for _, rk, _, _ in ab_only]
    dsn = [(df_ab.loc[rk, "Full_Sn_mean"] * 100.0 - full_sn) for _, rk, _, _ in ab_only]

    h_b = 0.34
    b_mcc = axB.barh(y_b + h_b / 2, dmcc, height=h_b, color=NAVY, edgecolor="#0F172A", lw=0.6,
                     label="$\\Delta\\mathrm{MCC} \\times 100$")
    b_sn = axB.barh(y_b - h_b / 2, dsn, height=h_b, color=CRIMSON, edgecolor="#7F1D1D", lw=0.6,
                    label="$\\Delta\\mathrm{Sensitivity}$ (pp)")
    for yv, vm, vs in zip(y_b, dmcc, dsn):
        axB.text(vm - 0.35, yv + h_b / 2, f"{vm / 100.0:+.4f}", ha="right", va="center", fontsize=7.3, fontweight="bold", color=NAVY)
        axB.text(vs - 0.35, yv - h_b / 2, f"{vs:+.2f} pp", ha="right", va="center", fontsize=7.3, fontweight="bold", color=CRIMSON)
    axB.axvline(0, color="#0F172A", lw=0.9)
    axB.set_yticks(y_b)
    axB.set_yticklabels(["no_coord\n(Static GNN)", "no_attn\n(Mean Fusion)", "no_3d\n(1D ESM-2 Only)", "no_esm\n(3D EGNN Only)"], fontsize=7.8)
    axB.set_xlim(-22.2, 1.5)
    axB.set_xlabel("Performance Change Relative to Full NeuroEGNN")
    axB.set_title("(B) Module Removal Impact ($\\Delta\\mathrm{MCC}$ & $\\Delta\\mathrm{Sn}$)", loc="left", fontweight="bold", pad=8)
    axB.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.5)
    axB.grid(axis="x", ls="--", alpha=0.28)
    despine(axB)

    # (C) Length-stratified sensitivity across all 5 variants
    axC = fig.add_subplot(gs[1, 0])
    bins_order = [
        ("Short (<=15 aa)", "Short ($\\leq 15$ aa)\n$n=368$ (209+/159-)"),
        ("Medium (16-30 aa)", "Medium ($16\\text{–}30$ aa)\n$n=330$ (133+/197-)"),
        ("Long (>30 aa)", "Long ($>30$ aa)\n$n=190$ (100+/90-)"),
    ]
    x_c = np.arange(len(bins_order))
    w_c = 0.15
    for v_i, (vk, _, v_lbl, v_col) in enumerate(variants):
        means, stds = [], []
        for bk, _ in bins_order:
            sv = df_len[(df_len["Length_Bin"] == bk) & (df_len["Variant"] == vk)]["Sensitivity"] * 100.0
            means.append(sv.mean())
            stds.append(sv.std(ddof=1))
        axC.bar(x_c + (v_i - 2) * w_c, means, w_c, yerr=stds, capsize=2.0,
                color=v_col, edgecolor="#1E293B", lw=0.55, label=vk)
    axC.annotate("+12.28 pp Sn\nvs. no_3d!", xy=(1 - 2 * w_c, 94.2), xytext=(0.55, 98.2),
                 fontsize=7.6, fontweight="bold", color=CRIMSON,
                 arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.1))
    axC.set_ylim(65, 103.5)
    axC.set_xticks(x_c)
    axC.set_xticklabels([bl for _, bl in bins_order], fontsize=7.8)
    axC.set_ylabel("Sensitivity / Recall (%, Mean ± SD)")
    axC.set_title("(C) Length-Stratified Sensitivity Across Regimes", loc="left", fontweight="bold", pad=20)
    axC.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=5, frameon=False, fontsize=7.3)
    axC.grid(axis="y", ls="--", alpha=0.28)
    despine(axC)

    # (D) Precision-Sensitivity trade-off across random seeds with iso-F1 contours
    axD = fig.add_subplot(gs[1, 1])
    sn_grid = np.linspace(78, 98, 200)
    pr_grid = np.linspace(77, 97, 200)
    SN_G, PR_G = np.meshgrid(sn_grid, pr_grid)
    F1_G = 2 * (SN_G * PR_G) / (SN_G + PR_G)
    cs = axD.contour(SN_G, PR_G, F1_G, levels=[84, 88, 90, 92, 94], colors="#CBD5E1", linestyles="--", linewidths=0.9)
    axD.clabel(cs, fmt="F1=%.0f%%", fontsize=6.8, colors="#64748B")

    y_te = pd.read_csv(os.path.join(CODE_DIR, "testing_v2.csv"))["Label"].values
    from sklearn.metrics import precision_score, recall_score, f1_score
    markers_d = {"full": "o", "no_coord": "s", "no_attn": "D", "no_3d": "^", "no_esm": "v"}
    for vk, _, _, v_col in variants:
        sns, prs, f1s = [], [], []
        for sd in (500, 501, 502):
            prob = pd.read_csv(os.path.join(CODE_DIR, "results", "ablation", vk, f"seed_{sd}", "test", "test_predictions.csv"))["Prediction_Probability"].values
            pred = (prob >= 0.5).astype(int)
            sns.append(recall_score(y_te, pred) * 100.0)
            prs.append(precision_score(y_te, pred) * 100.0)
            f1s.append(f1_score(y_te, pred) * 100.0)  # per-seed F1, averaged below (matches Table 5)
        axD.scatter(sns, prs, color=v_col, marker=markers_d[vk], s=36, alpha=0.55, edgecolor="white", lw=0.5)
        axD.errorbar(np.mean(sns), np.mean(prs), xerr=np.std(sns, ddof=1), yerr=np.std(prs, ddof=1),
                     fmt=markers_d[vk], color=v_col, markersize=8.5, markeredgecolor="#0F172A", markeredgewidth=0.9,
                     capsize=2.5, lw=1.2, label=f"{vk} (F1 = {np.mean(f1s):.1f}%)")
    axD.set_xlim(78.5, 97.5)
    axD.set_ylim(78.0, 96.5)
    axD.set_xlabel("Sensitivity / Recall (%)")
    axD.set_ylabel("Precision (%)")
    axD.set_title("(D) Precision–Sensitivity Operating Trade-Off (Seeds 500–502)", loc="left", fontweight="bold", pad=8)
    axD.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, fontsize=7.3,
               title="3-seed mean F1", title_fontsize=7.3)
    axD.grid(True, ls=":", alpha=0.25)
    despine(axD)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig3_ablation_and_length.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ===========================================================================
# 4. Figure 5: fig4_global_interpretability.pdf / .png (100% Zero-Leakage OOF N=8,887)
# ===========================================================================
def make_fig4_global_interp():
    print("[4/7] Generating fig4_global_interpretability ...")
    fig = plt.figure(figsize=(12.8, 9.0))
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.35)

    # (A) Spatial vs sequential pairwise attention regimes
    axA = fig.add_subplot(gs[0, 0])
    regimes = [
        "Local Sequence\n($1 \\leq |i-j| \\leq 2$)",
        "Medium 3D Contact\n($3 \\leq |i-j| \\leq 5, <10\\,\\mathrm{\\AA}$)",
        "Long-Range 3D Contact\n($|i-j| > 5, <10\\,\\mathrm{\\AA}$)",
    ]
    s2g_m = [np.mean(OOF_INTERP["s2g_local"]), np.mean(OOF_INTERP["s2g_med"]), np.mean(OOF_INTERP["s2g_long"])]
    s2g_s = [np.std(OOF_INTERP["s2g_local"]), np.std(OOF_INTERP["s2g_med"]), np.std(OOF_INTERP["s2g_long"])]
    g2s_m = [np.mean(OOF_INTERP["g2s_local"]), np.mean(OOF_INTERP["g2s_med"]), np.mean(OOF_INTERP["g2s_long"])]
    g2s_s = [np.std(OOF_INTERP["g2s_local"]), np.std(OOF_INTERP["g2s_med"]), np.std(OOF_INTERP["g2s_long"])]

    x_a = np.arange(3)
    w_a = 0.34
    b1 = axA.bar(x_a - w_a / 2, s2g_m, w_a, yerr=s2g_s, capsize=3.0, color=CRIMSON, edgecolor="#7F1D1D", lw=0.7,
                 label="1D $\\to$ 3D ($\\mathbf{A}_{\\mathrm{s}\\to\\mathrm{g}}$)")
    b2 = axA.bar(x_a + w_a / 2, g2s_m, w_a, yerr=g2s_s, capsize=3.0, color=STEEL, edgecolor=NAVY, lw=0.7,
                 label="3D $\\to$ 1D ($\\mathbf{A}_{\\mathrm{g}\\to\\mathrm{s}}$)")
    axA.axhline(1.0, color="#334155", ls="--", lw=1.0, label="Uniform Baseline ($1.0\\times$)")
    for rect, v in zip(b1, s2g_m):
        axA.text(rect.get_x() + rect.get_width() / 2, 0.76, f"{v:.3f}", ha="center", va="center",
                 fontsize=7.5, fontweight="bold", color="white")
    axA.annotate("+25.9% over Local", xy=(2 - w_a / 2, s2g_m[2] + s2g_s[2] + 0.01), xytext=(0.90, 1.40),
                 fontsize=8.0, fontweight="bold", color=CRIMSON,
                 arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.2))
    axA.set_xticks(x_a)
    axA.set_xticklabels(regimes, fontsize=7.8)
    axA.set_ylim(0.65, 1.50)
    axA.set_ylabel("Length-Normalized Cross-Attention ($L \\cdot \\bar{A}_{ij}$)")
    axA.set_title("(A) Pairwise Cross-Attention Across Spatial Contact Regimes", loc="left", fontweight="bold", pad=8)
    axA.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.4)
    axA.grid(axis="y", ls="--", alpha=0.28)
    despine(axA)

    # (B) Residue-type marginal attention enrichment (TP vs TN)
    axB = fig.add_subplot(gs[0, 1])
    aa_names = [str(a) for a in OOF_INTERP["aa_names"]]
    tp_m = OOF_INTERP["aa_tp_means"]
    tn_m = OOF_INTERP["aa_tn_means"]
    order_b = np.argsort(tp_m)[::-1]
    aa_ord = [aa_names[i] for i in order_b]
    tp_ord = tp_m[order_b]
    tn_ord = tn_m[order_b]

    x_b = np.arange(len(aa_ord))
    w_b = 0.38
    axB.bar(x_b - w_b / 2, tp_ord, w_b, color=CRIMSON, edgecolor="#7F1D1D", lw=0.5, label="True Positives (TP)")
    axB.bar(x_b + w_b / 2, tn_ord, w_b, color=STEEL, edgecolor=NAVY, lw=0.5, label="True Negatives (TN)")
    axB.axhline(1.0, color="#334155", ls="--", lw=0.9)
    m_pos = aa_ord.index("M")
    axB.annotate("TN $\\mathrm{Met} = 3.74\\times$\n($\\mathrm{Met}_1 = 5.27\\times$)",
                 xy=(m_pos + w_b / 2, tn_ord[m_pos]), xytext=(m_pos + 3.0, 3.25),
                 fontsize=7.6, fontweight="bold", color=NAVY,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor=STEEL, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.1))
    axB.set_xticks(x_b)
    axB.set_xticklabels(aa_ord, fontweight="bold", fontsize=7.8)
    axB.set_ylim(0, 4.25)
    axB.set_xlabel("Amino Acid Residue (Sorted by TP Normalized Attention)")
    axB.set_ylabel("Normalized Marginal Attention ($\\times$ Uniform)")
    axB.set_title("(B) Residue-Level Cross-Attention Enrichment ($N = 8,887$ OOF)", loc="left", fontweight="bold", pad=8)
    axB.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.4)
    axB.grid(axis="y", ls="--", alpha=0.28)
    despine(axB)

    # (C) Relative N-to-C positional attention profile
    axC = fig.add_subplot(gs[1, 0])
    rel_x = (np.arange(20) + 0.5) / 20.0
    tp_rel = OOF_INTERP["rel_tp_means"]
    tp_sem = OOF_INTERP["rel_tp_sems"]
    tn_rel = OOF_INTERP["rel_tn_means"]
    tn_sem = OOF_INTERP["rel_tn_sems"]

    axC.plot(rel_x, tp_rel, "-o", color=CRIMSON, lw=2.1, markersize=4.5, label="True Positives ($n = 4,144$)")
    axC.fill_between(rel_x, tp_rel - 1.96 * tp_sem, tp_rel + 1.96 * tp_sem, color=CRIMSON, alpha=0.18)
    axC.plot(rel_x, tn_rel, "-s", color=STEEL, lw=2.0, markersize=4.2, label="True Negatives ($n = 4,035$)")
    axC.fill_between(rel_x, tn_rel - 1.96 * tn_sem, tn_rel + 1.96 * tn_sem, color=STEEL, alpha=0.18)
    axC.axhline(1.0, color="#334155", ls="--", lw=0.9)
    axC.annotate(f"N-term $\\mathrm{{Met}}_1$ Rejection Spike\nin True Negatives (first bin, ${tn_rel[0]:.2f}\\times$)",
                 xy=(rel_x[0], tn_rel[0]), xytext=(0.18, 3.25),
                 fontsize=7.6, fontweight="bold", color=NAVY,
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.1))
    i_pk = int(np.argmax(tp_rel))
    axC.annotate(f"C-Terminal GPCR Pharmacophore Peak\nin True Positives (bin maximum, ${tp_rel[i_pk]:.2f}\\times$)",
                 xy=(rel_x[i_pk], tp_rel[i_pk]), xytext=(0.30, 2.15),
                 fontsize=7.6, fontweight="bold", color=CRIMSON,
                 arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.1))
    axC.set_xlim(0.0, 1.0)
    axC.set_ylim(0.45, 3.95)
    axC.set_xlabel("Normalized Relative Sequence Position ($0.0 = \\mathrm{N\\text{-}term} \\to 1.0 = \\mathrm{C\\text{-}term}$)")
    axC.set_ylabel("Normalized Attention ($\\times$ Uniform)")
    axC.set_title("(C) Polarized N-to-C Positional Attention Profile", loc="left", fontweight="bold", pad=8)
    axC.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.4)
    axC.grid(True, ls="--", alpha=0.28)
    despine(axC)

    # (D) Out-of-Fold Probability Calibration Histogram (N=8,887)
    axD = fig.add_subplot(gs[1, 1])
    oof_p = OOF_NPZ["oof_NeuroEGNN"]
    oof_y = OOF_NPZ["oof_y"]
    bins_d = np.linspace(0, 1, 26)
    axD.hist(oof_p[oof_y == 0], bins=bins_d, alpha=0.72, color=STEEL, edgecolor=NAVY, lw=0.6,
             label="Ground-Truth Negatives ($n = 4,463$)")
    axD.hist(oof_p[oof_y == 1], bins=bins_d, alpha=0.72, color=CRIMSON, edgecolor="#7F1D1D", lw=0.6,
             label="Ground-Truth Neuropeptides ($n = 4,424$)")
    axD.axvline(0.50, color="#0F172A", ls="--", lw=1.1, label="Decision Boundary ($P = 0.50$)")
    axD.axvspan(0.40, 0.60, color="#FEF3C7", alpha=0.55)
    axD.set_yscale("log")
    axD.set_ylim(3, 4500)
    axD.text(0.50, 900, "77.7% ($6,901 / 8,887$) High-Confidence\n($P \\geq 0.90$ or $P \\leq 0.10$)\nOnly 3.4% ($302$) in $[0.40, 0.60]$",
             ha="center", va="top", fontsize=7.6, fontweight="bold", color="#1E293B",
             bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#CBD5E1", lw=0.9))
    axD.set_xlabel("Out-of-Fold Predicted Neuropeptide Probability $P(\\mathrm{NP})$")
    axD.set_ylabel("Peptide Count (Log Scale)")
    axD.set_title("(D) Out-of-Fold Probability Calibration ($N = 8,887$)", loc="left", fontweight="bold", pad=8)
    axD.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), frameon=True, facecolor="white", framealpha=0.92, fontsize=7.1)
    axD.grid(axis="y", ls="--", alpha=0.28)
    despine(axD)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig4_global_interpretability.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ===========================================================================
# 5. Figure 6: fig5_case_studies_attention.pdf / .png (Zero-Leakage seed_500)
# ===========================================================================
def make_fig5_cases():
    print("[5/7] Generating fig5_case_studies_attention ...")
    df_test = pd.read_csv(os.path.join(CODE_DIR, "testing_v2.csv"))
    s2g_all = OOF_NPZ["s2g_s500"]
    g2s_all = OOF_NPZ["g2s_s500"]
    probs = OOF_NPZ["ho_egnn_s500"]

    fig = plt.figure(figsize=(13.2, 8.8))
    outer = fig.add_gridspec(2, 1, hspace=0.36)

    cases = [
        (608, "(A) True Positive Test-#608 (GIVDQCCTNICSRNQLMSYCND, $L=22$, $P(\\mathrm{NP})=0.9842$ — Disulfide-Bridged Neuropeptide)",
         [5, 6, 10, 19], TEAL, "Cys Disulfide Anchors (1.97× Uniform, 35.8% Share)"),
        (156, "(B) False Negative Test-#156 (MQIYGRPATMDE, $L=12$, $P(\\mathrm{NP})=0.1003 \\to 0.9187$ After Masking $\\mathrm{Met}_1$)",
         [0], CRIMSON, "N-Terminal $\\mathrm{Met}_1$ Suppressor Sink (46.8% $\\mathbf{A}_{\\mathrm{s}\\to\\mathrm{g}}$, 3.57× Uniform)"),
    ]

    for row_idx, (tid, title_str, hi_indices, hi_col, hi_lbl) in enumerate(cases):
        seq = str(df_test.iloc[tid]["Seq"]).strip()[:50]
        L = len(seq)
        s2g = s2g_all[tid, :L, :L] * L
        g2s = g2s_all[tid, :L, :L] * L
        s2g_m = s2g.mean(axis=0)
        g2s_m = g2s.mean(axis=0)
        comb = 0.5 * (s2g_m + g2s_m)
        comb_norm = comb / np.mean(comb)

        sub_gs = outer[row_idx].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.25], wspace=0.28)
        ax1 = fig.add_subplot(sub_gs[0, 0])
        ax2 = fig.add_subplot(sub_gs[0, 1])
        ax3 = fig.add_subplot(sub_gs[0, 2])

        res_ticks = [f"{ch}{i+1}" for i, ch in enumerate(seq)]
        vmax = min(5.5, max(np.percentile(s2g, 99), np.percentile(g2s, 99), 2.5))

        im1 = ax1.imshow(s2g, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto")
        ax1.set_title("1D $\\to$ 3D ($\\mathbf{A}_{\\mathrm{s}\\to\\mathrm{g}}$, $\\times$ Uniform)", fontsize=8.4, fontweight="bold")
        ax1.set_xlabel("3D Structure Key Residue $j$")
        ax1.set_ylabel("1D Sequence Query Residue $i$")

        im2 = ax2.imshow(g2s, cmap="YlGnBu", vmin=0.0, vmax=vmax, aspect="auto")
        ax2.set_title("3D $\\to$ 1D ($\\mathbf{A}_{\\mathrm{g}\\to\\mathrm{s}}$, $\\times$ Uniform)", fontsize=8.4, fontweight="bold")
        ax2.set_xlabel("1D Sequence Key Residue $j$")
        ax2.set_ylabel("3D Structure Query Residue $i$")

        step = 1 if L <= 14 else 2
        for ax_h, im in [(ax1, im1), (ax2, im2)]:
            ax_h.set_xticks(np.arange(0, L, step))
            ax_h.set_xticklabels([res_ticks[k] for k in range(0, L, step)], rotation=45, ha="right", fontsize=6.8)
            ax_h.set_yticks(np.arange(0, L, step))
            ax_h.set_yticklabels([res_ticks[k] for k in range(0, L, step)], fontsize=6.8)
            plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)

        # Marginal bar chart
        x_r = np.arange(L)
        bar_cols = [hi_col if idx in hi_indices else "#94A3B8" for idx in range(L)]
        bars = ax3.bar(x_r, comb_norm, color=bar_cols, edgecolor="#1E293B", lw=0.6, width=0.68)
        ax3.axhline(1.0, color="#0F172A", ls="--", lw=0.9, label="Uniform Expectation ($1.0\\times$)")
        for idx in hi_indices:
            ax3.text(idx, comb_norm[idx] + 0.07, f"{comb_norm[idx]:.2f}×", ha="center", va="bottom",
                     fontsize=7.3, fontweight="bold", color=hi_col)
        # Also label Arg6 on Test-#156 and Arg13 on Test-#608
        extra_idx = 12 if tid == 608 else 5
        ax3.text(extra_idx, comb_norm[extra_idx] + 0.07, f"{comb_norm[extra_idx]:.2f}×", ha="center", va="bottom",
                 fontsize=7.0, fontweight="bold", color=NAVY)
        bars[extra_idx].set_facecolor(STEEL)

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=hi_col, edgecolor="#1E293B", label=hi_lbl),
            Patch(facecolor=STEEL, edgecolor="#1E293B", label=f"Internal Basic Motif ({res_ticks[extra_idx]})"),
        ]
        ax3.set_xticks(x_r)
        ax3.set_xticklabels(res_ticks, rotation=45, ha="right", fontsize=7.0, fontweight="bold")
        ax3.set_ylim(0, max(comb_norm) * 1.28)
        ax3.set_ylabel("Marginal Attribution ($\\times$ Uniform)")
        ax3.set_title("Combined Residue Attribution Profile", fontsize=8.4, fontweight="bold")
        ax3.legend(handles=legend_handles, loc="upper right", fontsize=6.9, frameon=True, facecolor="white", framealpha=0.92)
        ax3.grid(axis="y", ls="--", alpha=0.28)
        despine(ax3)

        ax1.text(-0.18, 1.15, title_str, transform=ax1.transAxes, fontsize=9.5, fontweight="bold", color="#0F172A")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig5_case_studies_attention.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ===========================================================================
# 6. Figure 7: fig6_structural_relaxation_and_debiased.pdf / .png
# ===========================================================================
def make_fig6_relax_debiased():
    print("[6/7] Generating fig6_structural_relaxation_and_debiased ...")
    df_hs = pd.read_csv(os.path.join(CODE_DIR, "results", "debiased_and_relaxation", "holdout_seeds_sota_and_non_met1.csv")).set_index("Method")
    df_sep = pd.read_csv(os.path.join(CODE_DIR, "results", "debiased_and_relaxation", "latent_separability_metrics.csv")).set_index("Representation")

    bd = load_batch_tensors(
        os.path.join(CODE_DIR, "testing_v2.csv"),
        os.path.join(CODE_DIR, "cache", "esm", "testing_v2_t33_650M_L50.npz")
    )
    ck_full = load_ckpt(os.path.join(CODE_DIR, "results", "ablation", "full", "seed_500", "train", "best_model.pth"))
    ck_no3d = load_ckpt(os.path.join(CODE_DIR, "results", "ablation", "no_3d", "seed_500", "train", "best_model.pth"))
    out_full = forward_neuroegnn_numpy(ck_full["state_dict"], ck_full["config"], bd, return_details=True)
    out_no3d = forward_neuroegnn_numpy(ck_no3d["state_dict"], ck_no3d["config"], bd, return_details=True)

    mask_cf = bd["mask"].copy()
    mask_cf[:, 0] = 0.0
    probs_cf = forward_neuroegnn_numpy(ck_full["state_dict"], ck_full["config"], bd, mask_override=mask_cf)

    # Compute Rg-normalized trajectory stats (L >= 9)
    ct = out_full["coord_traj"]
    y_te = bd["labels"]
    pr_te = (out_full["probs"] >= 0.5).astype(int)
    rmsd_m, rmsd_med, rmsd_q25, rmsd_q75 = [], [], [], []
    tp_dnc, tn_dnc = [], []
    for l in range(5):
        r_list, tp_l, tn_l = [], [], []
        for i in range(len(y_te)):
            L = int(bd["lengths"][i])
            if L < 9:
                continue
            k = min(3, L // 3)
            x0 = ct[0][i, :L]
            c0 = x0 - x0.mean(axis=0, keepdims=True)
            rg0 = np.sqrt(np.mean(np.sum(c0 ** 2, axis=-1))) + 1e-8
            xl = ct[l][i, :L]
            cl = xl - xl.mean(axis=0, keepdims=True)
            rgl = np.sqrt(np.mean(np.sum(cl ** 2, axis=-1))) + 1e-8
            x_rg = cl * (rg0 / rgl)
            r_list.append(np.sqrt(np.mean(np.sum((x_rg - c0) ** 2, axis=-1))))
            d0 = np.linalg.norm(c0[:k].mean(axis=0) - c0[-k:].mean(axis=0))
            dl = np.linalg.norm(x_rg[:k].mean(axis=0) - x_rg[-k:].mean(axis=0))
            if y_te[i] == 1 and pr_te[i] == 1:
                tp_l.append((dl - d0) * 1000.0)
            elif y_te[i] == 0 and pr_te[i] == 0:
                tn_l.append((dl - d0) * 1000.0)
        rmsd_m.append(np.mean(r_list))
        rmsd_med.append(np.median(r_list))
        rmsd_q25.append(np.percentile(r_list, 25))
        rmsd_q75.append(np.percentile(r_list, 75))
        tp_dnc.append(np.mean(tp_l))
        tn_dnc.append(np.mean(tn_l))

    fig = plt.figure(figsize=(13.2, 9.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], wspace=0.26, hspace=0.34)

    # (A) Non-Met1 Hard Subset Benchmark (n = 602: 436 Pos / 166 Neg)
    axA = fig.add_subplot(gs[0, 0])
    m_order = [
        ("NeuroPred-CLQ (Bioinf 2022)", "NeuroPred-CLQ\n(2022)"),
        ("UniDL4BioPep (BiB 2023)", "UniDL4BioPep\n(2023)"),
        ("NeuroPred-PLM (BiB 2024)", "NeuroPred-PLM\n(2023)"),
        ("w/o 3D-EGNN (no_3d)", "1D ESM-2\n(no_3d)"),
        ("w/o Coord Update (no_coord)", "Static GNN\n(no_coord)"),
        ("NeuroEGNN (full)", "NeuroEGNN\n(Ours)"),
    ]
    x_a = np.arange(len(m_order))
    w_a = 0.25
    mcc_v = [df_hs.loc[mk, "NonM1_MCC_mean"] * 100 for mk, _ in m_order]
    mcc_e = [df_hs.loc[mk, "NonM1_MCC_std"] * 100 for mk, _ in m_order]
    sn_v = [df_hs.loc[mk, "NonM1_Sn_mean"] * 100 for mk, _ in m_order]
    sn_e = [df_hs.loc[mk, "NonM1_Sn_std"] * 100 for mk, _ in m_order]
    auc_v = [df_hs.loc[mk, "NonM1_AUC_mean"] * 100 for mk, _ in m_order]
    auc_e = [df_hs.loc[mk, "NonM1_AUC_std"] * 100 for mk, _ in m_order]

    axA.bar(x_a - w_a, mcc_v, w_a, yerr=mcc_e, capsize=2.2, color=NAVY, edgecolor="#0F172A", lw=0.6, label="MCC (×100)")
    axA.bar(x_a, sn_v, w_a, yerr=sn_e, capsize=2.2, color=CRIMSON, edgecolor="#7F1D1D", lw=0.6, label="Sensitivity (%)")
    axA.bar(x_a + w_a, auc_v, w_a, yerr=auc_e, capsize=2.2, color=TEAL, edgecolor="#115E59", lw=0.6, label="AUC-ROC (×100)")

    axA.annotate("Sn: 95.34% (+8.87 pp vs. no_3d)\n$\\Delta$MCC = +0.0681 vs. NeuroPred-PLM",
                 xy=(x_a[-1], sn_v[-1] + sn_e[-1] + 0.5), xytext=(2.15, 98.8),
                 fontsize=7.3, fontweight="bold", color=CRIMSON,
                 arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.1))
    axA.set_ylim(50, 105.0)
    axA.set_xticks(x_a)
    axA.set_xticklabels([sl for _, sl in m_order], fontsize=7.0)
    axA.set_ylabel("Non-$\\mathrm{Met}_1$ Subset Score")
    axA.set_title("(A) Debiased Non-$\\mathrm{Met}_1$ Hard Subset ($n = 602$, $\\mathrm{AA}_1 \\neq \\mathrm{M}$)", loc="left", fontweight="bold", pad=30)
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False, ncol=3, fontsize=7.3)
    axA.grid(axis="y", ls="--", alpha=0.28)
    despine(axA)

    # (B) Rg-Normalized E(3)-EGNN Coordinate Relaxation Trajectory (100% aligned with Table 9)
    axB = fig.add_subplot(gs[0, 1])
    layers = np.arange(5)
    l1 = axB.plot(layers, tp_dnc, "-o", color=CRIMSON, lw=2.2, markersize=5.8,
                  label="TP $\\Delta d_{\\mathrm{N-C}}$ ($-229.3\\,\\mathrm{m\\AA}$)")
    l2 = axB.plot(layers, tn_dnc, "--s", color=STEEL, lw=1.9, markersize=5.2,
                  label="TN $\\Delta d_{\\mathrm{N-C}}$ ($-148.6\\,\\mathrm{m\\AA}$)")
    axB.axhline(0, color="#334155", lw=0.7, alpha=0.5)
    axB.set_ylabel("N-to-C Inter-Domain Shift $\\Delta d_{\\mathrm{N-C}}$ ($\\mathrm{m\\AA}$)", color=CRIMSON)
    axB.set_ylim(-255, 25)

    axB2 = axB.twinx()
    axB2.fill_between(layers, rmsd_q25, rmsd_q75, color=TEAL, alpha=0.18, label="$\\mathrm{RMSD}_{R_g}$ IQR")
    l3 = axB2.plot(layers, rmsd_m, ":D", color="#0F766E", lw=2.0, markersize=5.2,
                   label="Mean $\\mathrm{RMSD}_{R_g}$ ($0.352\\,\\mathrm{\\AA}$)")
    l4 = axB2.plot(layers, rmsd_med, "-.^", color=NAVY, lw=1.6, markersize=4.8,
                   label="Median $\\mathrm{RMSD}_{R_g}$ ($0.038\\,\\mathrm{\\AA}$)")
    axB2.set_ylabel("$R_g$-Normalized Backbone $\\mathrm{RMSD}_{R_g}$ ($\\mathrm{\\AA}$)", color="#0F766E")
    axB2.set_ylim(-0.02, 0.45)
    axB2.spines["top"].set_visible(False)

    axB.set_xticks(layers)
    axB.set_xticklabels([f"Layer {l}\n($\\tilde{{\\mathbf{{X}}}}^{{({l})}}$)" for l in layers], fontsize=7.8)
    axB.set_title("(B) $R_g$-Normalized $E(3)$-EGNN Relaxation Trajectory", loc="left", fontweight="bold", pad=30)
    axB.grid(True, ls="--", alpha=0.28)
    axB.spines["top"].set_visible(False)

    handles_b = l1 + l2 + l3 + l4
    labels_b = [h.get_label() for h in handles_b]
    axB.legend(handles_b, labels_b, loc="upper center", bbox_to_anchor=(0.5, 1.135), ncol=2, fontsize=6.7, frameon=False)

    # (C) Counterfactual Met1 Masking Rescue (Generous bottom ylim so legend never touches Test-#837!)
    axC = fig.add_subplot(gs[1, 0])
    m1_idx = [i for i, (s, y) in enumerate(zip(bd["raw_seqs"], y_te)) if y == 1 and str(s).strip().startswith("M")]
    p_orig = out_full["probs"][m1_idx]
    p_cf = probs_cf[m1_idx]
    order_c = np.argsort(p_orig)
    y_c = np.arange(len(m1_idx))
    seq_lbls = [f"Test-#{m1_idx[k]} ({bd['raw_seqs'][m1_idx[k]][:9]}..)" for k in order_c]

    axC.axvline(0.50, color=CRIMSON, ls="--", lw=1.2, label="Decision Threshold ($P = 0.50$)")
    for k_pos, ord_k in enumerate(order_c):
        axC.plot([p_orig[ord_k], p_cf[ord_k]], [k_pos, k_pos], color=TEAL, lw=2.2, zorder=2)
        axC.text(max(p_orig[ord_k], p_cf[ord_k]) + 0.02, k_pos, f"{p_orig[ord_k]:.2f}$\\to${p_cf[ord_k]:.2f}",
                 va="center", ha="left", fontsize=7.0, fontweight="bold", color="#0F766E")
    axC.scatter(p_orig[order_c], y_c, color=AMBER, edgecolor="#7F2B1A", s=58, zorder=3,
                label="Original (with $\\mathrm{Met}_1$, Recall 16.7%)")
    axC.scatter(p_cf[order_c], y_c, color=TEAL, edgecolor="#0B3B2E", s=66, marker="D", zorder=4,
                label="$\\mathrm{Met}_1$ Masked Out (Rescued, Recall 83.3%)")
    axC.set_yticks(y_c)
    axC.set_yticklabels(seq_lbls, fontfamily="monospace", fontsize=7.4)
    axC.set_xlim(-0.03, 1.19)
    axC.set_ylim(-1.65, 5.65)
    axC.set_xlabel("Predicted Neuropeptide Probability $P(\\mathrm{NP})$")
    axC.set_title("(C) Counterfactual Causal Rescue on $\\mathrm{Met}_1$-Positives ($n = 6$)", loc="left", fontweight="bold", pad=8)
    axC.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=7.1)
    axC.grid(axis="x", ls="--", alpha=0.28)
    despine(axC)

    # (D) Latent Manifold Evolution (t-SNE)
    gsD = gs[1, 1].subgridspec(1, 2, wspace=0.12)
    axD1 = fig.add_subplot(gsD[0, 0])
    axD2 = fig.add_subplot(gsD[0, 1])

    tsne_1d = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto").fit_transform(out_no3d["pooled"])
    tsne_full = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto").fit_transform(out_full["pooled"])

    for ax_d, coords, sub_t, k_sep in [
        (axD1, tsne_1d, "1D ESM-2 Only (no_3d)", "1D ESM-2 Only (no_3d, 256-D)"),
        (axD2, tsne_full, "NeuroEGNN Co-Attn Fused", "NeuroEGNN Co-Attn Fused (256-D)"),
    ]:
        ax_d.scatter(coords[y_te == 0, 0], coords[y_te == 0, 1], c=STEEL, s=10, alpha=0.55, label="Non-Neuropeptide", rasterized=True)
        ax_d.scatter(coords[y_te == 1, 0], coords[y_te == 1, 1], c=CRIMSON, s=10, alpha=0.65, label="Neuropeptide", rasterized=True)
        sil = df_sep.loc[k_sep, "Silhouette_Cosine"]
        fdr = df_sep.loc[k_sep, "Fisher_Discriminant_Ratio"]
        ax_d.set_title(f"{sub_t}\nSil = {sil:.3f} | FDR = {fdr:.2f}", fontsize=8.0, fontweight="bold")
        ax_d.set_xticks([])
        ax_d.set_yticks([])
        for sp in ax_d.spines.values():
            sp.set_color("#CBD5E1")
    axD1.text(-0.06, 1.14, "(D) Latent Manifold Separability ($t$-SNE, $N = 888$)",
              transform=axD1.transAxes, fontsize=9.6, fontweight="bold")
    axD2.legend(loc="lower right", markerscale=1.8, fontsize=7.0, frameon=True, facecolor="white", framealpha=0.92)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig6_structural_relaxation_and_debiased.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ===========================================================================
# 7. Table of Contents (TOC) Graphic: exact 3.25 in x 1.75 in (<= 8.25 x 4.45 cm)
# ===========================================================================
def make_toc_graphic():
    print("[7/7] Generating fig_toc_graphic (exact 3.25 x 1.75 in) ...")
    fig = plt.figure(figsize=(3.25, 1.75), dpi=600)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 325)
    ax.set_ylim(0, 175)
    ax.axis("off")

    # Outer card
    ax.add_patch(FancyBboxPatch((2, 2), 321, 171, boxstyle="round,pad=0,rounding_size=8",
                                facecolor="#F8FAFC", edgecolor="#94A3B8", lw=1.0))

    # Header bar
    ax.add_patch(FancyBboxPatch((6, 146), 313, 23, boxstyle="round,pad=0,rounding_size=5",
                                facecolor=NAVY, edgecolor="none"))
    ax.text(162.5, 157.5, "NeuroEGNN: Equivariant 3D Co-Attention",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="white")

    # Left panel: Sequence + 3D Peptide Graph visual
    ax.add_patch(FancyBboxPatch((6, 8), 96, 132, boxstyle="round,pad=0,rounding_size=6",
                                facecolor="#EFF6FF", edgecolor="#60A5FA", lw=0.8))
    ax.text(54, 130, "Peptide Input", ha="center", va="center", fontsize=7.0, fontweight="bold", color=NAVY)
    ax.text(54, 115, "1D: -F-A-P-R-L-a", ha="center", va="center", fontsize=6.4, fontweight="bold",
            color="#1E40AF", bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#93C5FD", lw=0.6))

    # Mini 3D peptide backbone graph node-link illustration
    pts = [(22, 82), (38, 94), (62, 92), (78, 72), (64, 50), (36, 54)]
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
        ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], color=NAVY, lw=1.5, zorder=2)
    # Non-local 3D contact dashed lines
    for u, v in [(1, 4), (0, 5), (2, 5)]:
        ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], color=CRIMSON, ls="--", lw=1.0, zorder=2)
    node_cols = [STEEL, TEAL, TEAL, CRIMSON, CRIMSON, AMBER]
    for (px, py), nc in zip(pts, node_cols):
        ax.add_patch(Circle((px, py), radius=5.2, facecolor=nc, edgecolor="white", lw=0.8, zorder=3))
    ax.text(54, 31, "3D Contact Graph", ha="center", va="center", fontsize=6.5, fontweight="bold", color="#0F766E")
    ax.text(54, 18, "Homology < 80%", ha="center", va="center", fontsize=6.2, color="#475569", fontweight="bold")

    # Center panel: Dual-Stream Fusion
    ax.add_patch(FancyBboxPatch((111, 8), 106, 132, boxstyle="round,pad=0,rounding_size=6",
                                facecolor="#FFFBEB", edgecolor="#F59E0B", lw=0.85))
    ax.text(164, 131, "Dual-Modal Fusion", ha="center", va="center", fontsize=7.0, fontweight="bold", color="#92400E")

    ax.add_patch(FancyBboxPatch((116, 98), 96, 24, boxstyle="round,pad=0,rounding_size=4",
                                facecolor="#DBEAFE", edgecolor="#2563EB", lw=0.7))
    ax.text(164, 110, "ESM-2 (650M)\n1D Semantics", ha="center", va="center", fontsize=6.0, fontweight="bold", color=NAVY)

    ax.add_patch(FancyBboxPatch((116, 63), 96, 18, boxstyle="round,pad=0,rounding_size=4",
                                facecolor="#FEE2E2", edgecolor=CRIMSON, lw=0.75))
    ax.text(164, 72, "Gated Co-Attention", ha="center", va="center", fontsize=5.8, fontweight="bold", color="#991B1B")

    ax.add_patch(FancyBboxPatch((116, 14), 96, 30, boxstyle="round,pad=0,rounding_size=4",
                                facecolor="#CCFBF1", edgecolor="#0D9488", lw=0.7))
    ax.text(164, 29, "4-Layer E(3)-EGNN\n3D Force Relaxation", ha="center", va="center", fontsize=5.8, fontweight="bold", color="#115E59")

    # Bidirectional vertical arrows strictly between boxes
    ax.annotate("", xy=(164, 82), xytext=(164, 97), arrowprops=dict(arrowstyle="<|-|>", lw=0.9, color=CRIMSON,
                mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(164, 45), xytext=(164, 62), arrowprops=dict(arrowstyle="<|-|>", lw=0.9, color="#0D9488",
                mutation_scale=5, shrinkA=0, shrinkB=0))

    # Right panel: Compact Performance Bars + Key Metrics
    ax.add_patch(FancyBboxPatch((226, 8), 93, 132, boxstyle="round,pad=0,rounding_size=6",
                                facecolor="#ECFDF5", edgecolor="#10B981", lw=0.85))
    ax.text(272.5, 131, "5-Fold CV MCC", ha="center", va="center", fontsize=7.0, fontweight="bold", color="#065F46")

    # Mini horizontal bar comparison (5-fold CV MCC)
    bar_y = [106, 90, 74]
    bar_w = [80, 71, 64]
    bar_c = [CRIMSON, NAVY, SLATE]
    bar_t = ["Ours  .842", "UniDL .816", "PLM   .798"]
    for by, bw, bc, bt in zip(bar_y, bar_w, bar_c, bar_t):
        ax.add_patch(FancyBboxPatch((232, by - 5.2), bw, 10.4, boxstyle="round,pad=0,rounding_size=2",
                                    facecolor=bc, edgecolor="none", alpha=0.92))
        ax.text(234.5, by, bt, ha="left", va="center", fontsize=5.7, fontweight="bold", color="white")

    ax.text(272.5, 52, "AUC: 0.978", ha="center", va="center", fontsize=7.0, fontweight="bold", color="#065F46")
    ax.text(272.5, 37, "Sn: 93.7%", ha="center", va="center", fontsize=6.5, fontweight="bold", color=CRIMSON)
    ax.text(272.5, 21, "C-term GPCR Motif", ha="center", va="center", fontsize=5.6, fontweight="bold", color="#0F766E")

    # Horizontal flow arrows
    ax.annotate("", xy=(111, 74), xytext=(102, 74), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#334155",
                mutation_scale=6, shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(226, 74), xytext=(217, 74), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#059669",
                mutation_scale=6, shrinkA=0, shrinkB=0))

    fig.savefig(os.path.join(FIG_DIR, "fig_toc_graphic.pdf"), dpi=600)
    fig.savefig(os.path.join(FIG_DIR, "fig_toc_graphic.png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    _all = {"1": make_fig1_dataset, "2": make_fig2_performance, "3": make_fig3_ablation,
            "4": make_fig4_global_interp, "5": make_fig5_cases, "6": make_fig6_relax_debiased,
            "toc": make_toc_graphic}
    for _k in (sys.argv[1:] or list(_all)):
        _all[_k]()

    print("\n=== Verifying PDF Dimensions & Integrity ===")
    for fn in [
        "fig0_architecture.pdf",
        "fig1_dataset_analysis.pdf",
        "fig2_performance_comparison.pdf",
        "fig3_ablation_and_length.pdf",
        "fig4_global_interpretability.pdf",
        "fig5_case_studies_attention.pdf",
        "fig6_structural_relaxation_and_debiased.pdf",
        "fig_toc_graphic.pdf",
    ]:
        fp = os.path.join(FIG_DIR, fn)
        if not os.path.exists(fp):
            continue
        doc = fitz.open(fp)
        rect = doc[0].rect
        print(f"  {fn:45s}: {rect.width:6.1f} x {rect.height:6.1f} pt ({rect.width/72:.2f} x {rect.height/72:.2f} in)")
        doc.close()
