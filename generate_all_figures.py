# -*- coding: utf-8 -*-
"""Render benchmark, ablation, interpretability, relaxation, and TOC figures (Figures 2-6, 8, and TOC)."""
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
    "font.size": 9.0,
    "axes.labelsize": 9.8,
    "axes.titlesize": 13.0,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "legend.fontsize": 8.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})

# Color palette
NAVY = "#3D556E"
CRIMSON = "#9A5B5B"
TEAL = "#4D7A75"
STEEL = "#688299"
AMBER = "#A67C63"
GOLD = "#C2AE78"
PLUM = "#786982"
SLATE = "#6B7885"

BAR_ROSE = "#B88888"
BAR_BLUE = "#8298AD"
BAR_NAVY = "#64788C"
BAR_TEAL = "#7A9E99"
BAR_AMBER = "#BFA08A"
BAR_PLUM = "#94879E"
BAR_SLATE = "#98A2AC"
BAR_MUTED = "#C4CBD2"

ERR_KW = dict(ecolor="#334155", elinewidth=0.95, capthick=0.95)

with open(os.path.join(DATA_DIR, "clean_diagnostics.json")) as f:
    DIAG = json.load(f)
OOF_NPZ = np.load(os.path.join(DATA_DIR, "clean_oof_and_holdout.npz"), allow_pickle=True)
OOF_INTERP = np.load(os.path.join(DATA_DIR, "oof_interp_8887.npz"), allow_pickle=True)


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# Figure 2: fig1_dataset_analysis.pdf / .png
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

    fig = plt.figure(figsize=(13.2, 9.4))
    gs = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.39)

    # (A) Sequence length distribution
    axA = fig.add_subplot(gs[0, 0])
    bins = np.arange(5, 102, 3)
    axA.hist(neg_lens, bins=bins, density=True, alpha=0.45, color=BAR_BLUE, edgecolor="none", lw=0,
             label=f"Non-Neuropeptide ($n=4,463$, {neg_lens.mean():.1f}±{neg_lens.std():.1f} aa)")
    axA.hist(pos_lens, bins=bins, density=True, alpha=0.48, color=BAR_ROSE, edgecolor="none", lw=0,
             label=f"Neuropeptide ($n=4,424$, {pos_lens.mean():.1f}±{pos_lens.std():.1f} aa)")
    xs = np.linspace(6, 99, 300)
    axA.plot(xs, gaussian_kde(neg_lens, bw_method=0.22)(xs), color=NAVY, lw=1.8)
    axA.plot(xs, gaussian_kde(pos_lens, bw_method=0.22)(xs), color=CRIMSON, lw=2.0)
    axA.axvline(50, color=TEAL, ls="--", lw=1.5)
    axA.text(51.5, 0.066, "$L_{\\max} = 50$ aa\n(91.49% full-length)", color=TEAL,
             fontsize=8.0, fontweight="bold", va="top")
    axA.axvline(15.5, color="#94A3B8", ls=":", lw=1.0)
    axA.axvline(30.5, color="#94A3B8", ls=":", lw=1.0)
    axA.set_xlabel("Peptide Sequence Length $L$ (aa)")
    axA.set_ylabel("Probability Density")
    axA.set_xlim(4, 101)
    axA.set_ylim(0, 0.092)
    axA.set_title("(A) Sequence Length Distributions Across Classes", loc="left", fontweight="bold", pad=10)
    axA.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.8)
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
    colors_b = [BAR_BLUE] + [BAR_TEAL] * 7
    bars_b = axB.bar(x_b, bucket_counts, width=0.62, color=colors_b, edgecolor="none", lw=0)
    axB.set_yscale("log")
    axB.set_ylim(0.7, 16000)
    for rect, cnt in zip(bars_b, bucket_counts):
        axB.text(rect.get_x() + rect.get_width() / 2, cnt * 1.22, f"{cnt:,}",
                 ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#1E293B")
    axB.text(0.97, 0.88,
             "Total: $C = 7,050$ clusters ($\\tau = 0.80$)\n"
             "• 6,526 singletons (73.43% of peptides)\n"
             "• 524 multi-member families (2,361 peptides,\n"
             "  26.57% at risk under random splits; max = 191)",
             transform=axB.transAxes, ha="right", va="top", fontsize=7.8,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.9))
    axB.set_xticks(x_b)
    axB.set_xticklabels(bin_labels, fontsize=8.0)
    axB.set_xlabel("Cluster Cardinality $|\\mathcal{C}_c|$ (Sequences per Homology Family)")
    axB.set_ylabel("Number of Clusters (Log Scale)")
    axB.set_title("(B) Homology Cluster Distribution ($\\tau = 0.80$)", loc="left", fontweight="bold", pad=10)
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
    c_bars = [BAR_ROSE if v >= 0 else BAR_BLUE for v in fc_vals]
    y_c = np.arange(len(aa_sorted))
    axC.barh(y_c, fc_vals, height=0.68, color=c_bars, edgecolor="none", lw=0)
    axC.axvline(0, color="#334155", lw=0.9)
    axC.set_yticks(y_c)
    axC.set_yticklabels(aa_sorted, fontweight="bold", fontsize=8.0)
    axC.set_xlim(-1.18, 0.72)
    axC.set_xlabel("Global Compositional Log$_2$ Enrichment $\\log_2(f_{\\mathrm{Pos}} / f_{\\mathrm{Neg}})$")
    axC.set_title("(C) Residue-Level Compositional Enrichment", loc="left", fontweight="bold", pad=10)
    axC.text(-1.12, 15.2, "Enriched in Negatives\n(Ile, Met, Lys, Val)", fontsize=7.8, color=NAVY, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1", alpha=0.92))
    axC.text(0.06, 1.4, "Enriched in Neuropeptides\n(Pro, Gly, Asp, Glu, Trp, Phe)", fontsize=7.8, color=CRIMSON, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E5C2C2", alpha=0.92))
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
    axD.bar(positions - w_d / 2, neg_m_freq, width=w_d * 0.90, color=BAR_BLUE, edgecolor="none", lw=0,
            label="Negative Controls")
    axD.bar(positions + w_d / 2, pos_m_freq, width=w_d * 0.90, color=BAR_ROSE, edgecolor="none", lw=0,
            label="Mature Neuropeptides")
    axD.annotate("65.47% Neg vs. 1.99% Pos\n(32.9× N-terminal $\\mathrm{Met}_1$ disparity)",
                 xy=(1 - w_d / 2, 65.5), xytext=(3.3, 55.0),
                 fontsize=8.2, fontweight="bold", color=NAVY,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#FAF3F3", edgecolor=BAR_ROSE, lw=0.9),
                 arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.2))
    axD.set_xticks(positions)
    axD.set_ylim(0, 74)
    axD.set_xlabel("Residue Position from N-Terminus ($i = 1, \\dots, 15$)")
    axD.set_ylabel("Methionine Frequency at Position $i$ (%)")
    axD.set_title("(D) Positional N-Terminal $\\mathrm{Met}_1$ Cleavage Bias", loc="left", fontweight="bold", pad=10)
    axD.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92)
    axD.grid(axis="y", ls="--", alpha=0.28)
    despine(axD)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig1_dataset_analysis.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# Figure 3: fig2_performance_comparison.pdf / .png
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

    # Report 3-seed mean AUC and AUPRC in legend
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

    fig = plt.figure(figsize=(13.2, 9.6))
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.40)

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
    ax_ins.tick_params(labelsize=6.6)
    ax_ins.grid(True, ls=":", alpha=0.35)
    ax_ins.set_title("High-Sp Zoom (FPR $\\leq 0.20$)", fontsize=7.8, fontweight="bold", pad=3)
    axA.set_xlim(-0.01, 1.01)
    axA.set_ylim(-0.01, 1.02)
    axA.set_xlabel("False Positive Rate ($1 - \\mathrm{Specificity}$)")
    axA.set_ylabel("True Positive Rate (Sensitivity)")
    axA.set_title("(A) Hold-Out ROC Curves ($N = 888$)", loc="left", fontweight="bold", pad=10)
    axA.legend(loc="lower right", fontsize=6.7, labelspacing=0.28, frameon=True, facecolor="white", framealpha=0.95)
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
    axB.set_title("(B) Hold-Out Precision–Recall Curves ($N = 888$)", loc="left", fontweight="bold", pad=10)
    axB.legend(loc="lower left", fontsize=7.1, frameon=True, facecolor="white", framealpha=0.94)
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

    axC.bar(x_c - w_c, np.array(acc_v) * 100, w_c * 0.90, yerr=np.array(acc_e) * 100, capsize=2.0,
            error_kw=ERR_KW, color=BAR_BLUE, edgecolor="none", lw=0, label="Accuracy (%)")
    axC.bar(x_c, np.array(sn_v) * 100, w_c * 0.90, yerr=np.array(sn_e) * 100, capsize=2.0,
            error_kw=ERR_KW, color=BAR_ROSE, edgecolor="none", lw=0, label="Sensitivity (%)")
    axC.bar(x_c + w_c, np.array(auc_v) * 100, w_c * 0.90, yerr=np.array(auc_e) * 100, capsize=2.0,
            error_kw=ERR_KW, color=BAR_TEAL, edgecolor="none", lw=0, label="AUC-ROC (×100)")
    axC.set_ylim(78, 101.5)
    axC.set_xticks(x_c)
    axC.set_xticklabels([lbl for lbl, _, _ in cv_items], fontsize=7.5)
    axC.set_ylabel("5-Fold Cross-Validation Score (%)")
    axC.set_title("(C) 5-Fold Homology-Isolated CV ($N = 8,887$)", loc="left", fontweight="bold", pad=23)
    axC.legend(loc="upper center", bbox_to_anchor=(0.5, 1.115), ncol=3, frameon=False, fontsize=8.0)
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
        ("NeuroPred\n-FRL", np.array(frl_fold_mcc), BAR_PLUM),
        ("PredNeuroP", df_pf[df_pf["Method"] == "PredNeuroP (BiB 2020)"]["Full_MCC"].values, BAR_AMBER),
        ("ESM-2\n+LR", df_pf[df_pf["Method"] == "ESM-2 (650M) + LR"]["Full_MCC"].values, BAR_BLUE),
        ("NeuroPred\n-PLM", df_pf[df_pf["Method"] == "NeuroPred-PLM (BiB 2024)"]["Full_MCC"].values, BAR_NAVY),
        ("UniDL4\nBioPep", df_pf[df_pf["Method"] == "UniDL4BioPep (BiB 2023)"]["Full_MCC"].values, BAR_TEAL),
        ("NeuroEGNN\n(Ours)", df_pf[df_pf["Method"] == "NeuroEGNN"]["Full_MCC"].values, BAR_ROSE),
    ]
    pos_d = np.arange(len(d_methods))
    bp = axD.boxplot([vals for _, vals, _ in d_methods], positions=pos_d, widths=0.46,
                     patch_artist=True, showfliers=False)
    for patch, (_, _, col) in zip(bp["boxes"], d_methods):
        patch.set_facecolor(col)
        patch.set_alpha(0.36)
        patch.set_edgecolor(col)
        patch.set_linewidth(1.1)
    for med in bp["medians"]:
        med.set_color("#1E293B")
        med.set_linewidth(1.5)
    rng = np.random.RandomState(42)
    for i, (_, vals, col) in enumerate(d_methods):
        jit = rng.uniform(-0.11, 0.11, size=len(vals))
        axD.scatter(np.full_like(vals, i, dtype=float) + jit, vals, color=col, edgecolor="white", s=34, lw=0.6, zorder=4)

    # Significance brackets (OOF DeLong & McNemar)
    axD.plot([5, 5, 6, 6], [0.871, 0.878, 0.878, 0.871], color="#334155", lw=1.0)
    axD.text(5.5, 0.880, "McNemar $p=1.8\\times 10^{-5}$\nDeLong $p=4.1\\times 10^{-10}$",
             ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=CRIMSON)
    axD.plot([4, 4, 6, 6], [0.845, 0.910, 0.910, 0.903], color="#334155", lw=0.9)
    axD.text(5.0, 0.912, "vs. NeuroPred-PLM: McNemar $p=9.2\\times 10^{-13}$, DeLong $p=1.5\\times 10^{-17}$",
             ha="center", va="bottom", fontsize=6.7, fontweight="bold", color=NAVY)

    axD.set_xticks(pos_d)
    axD.set_xticklabels([m for m, _, _ in d_methods], fontsize=7.6)
    axD.set_ylim(0.70, 0.942)
    axD.set_ylabel("Fold-Wise MCC ($K = 5$ Folds)")
    axD.set_title("(D) Fold-Wise MCC & OOF Significance ($N = 8,887$)", loc="left", fontweight="bold", pad=10)
    axD.grid(axis="y", ls="--", alpha=0.28)
    despine(axD)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig2_performance_comparison.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# Figure 4: fig3_ablation_and_length.pdf / .png
def make_fig3_ablation():
    print("[3/7] Generating fig3_ablation_and_length ...")
    df_ab = pd.read_csv(os.path.join(CODE_DIR, "results", "debiased_and_relaxation", "holdout_seeds_sota_and_non_met1.csv")).set_index("Method")
    df_len = pd.read_csv(os.path.join(CODE_DIR, "results", "baselines", "ablation_length_stratified.csv"))

    variants = [
        ("full", "NeuroEGNN (full)", "NeuroEGNN\n(0.917M)", "NeuroEGNN", "NeuroEGNN", BAR_ROSE),
        ("no_coord", "w/o Coord Update (no_coord)", "w/o Coord.\n(0.851M)", "w/o Coord.", "w/o Coord. Update", BAR_AMBER),
        ("no_attn", "w/o Cross-Attn (no_attn)", "w/o Co-Attn\n(0.735M)", "w/o Co-Attn", "w/o Co-Attention", BAR_TEAL),
        ("no_3d", "w/o 3D-EGNN (no_3d)", "w/o 3D-EGNN\n(0.271M)", "w/o 3D-EGNN", "w/o 3D-EGNN", BAR_NAVY),
        ("no_esm", "w/o 1D ESM-2 (no_esm)", "w/o ESM-2\n(0.571M)", "w/o ESM-2", "w/o ESM-2", BAR_SLATE),
    ]

    fig = plt.figure(figsize=(13.2, 9.6))
    gs = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.42)

    # (A) Multi-metric bar chart with legend cleanly above bars
    axA = fig.add_subplot(gs[0, 0])
    x_a = np.arange(len(variants))
    w_a = 0.18
    metrics_a = [
        ("Full_Acc", "Acc (%)", BAR_BLUE, 100.0),
        ("Full_Sn", "Sn (%)", BAR_ROSE, 100.0),
        ("Full_Sp", "Sp (%)", BAR_PLUM, 100.0),
        ("Full_AUC", "AUC (×100)", BAR_TEAL, 100.0),
    ]
    for m_i, (col, lbl, c_bar, scale) in enumerate(metrics_a):
        vals = [df_ab.loc[rk, f"{col}_mean"] * scale for _, rk, _, _, _, _ in variants]
        errs = [df_ab.loc[rk, f"{col}_std"] * scale for _, rk, _, _, _, _ in variants]
        offset = (m_i - 1.5) * w_a
        axA.bar(x_a + offset, vals, w_a * 0.90, yerr=errs, capsize=2.0, error_kw=ERR_KW,
                color=c_bar, edgecolor="none", lw=0, label=lbl)
    axA.set_ylim(76, 100.8)
    axA.set_xticks(x_a)
    axA.set_xticklabels([xl for _, _, xl, _, _, _ in variants], fontsize=7.8)
    axA.set_ylabel("Hold-Out Score (%, Mean ± SD)")
    axA.set_title("(A) Architectural Ablation (3 Random Seeds)", loc="left", fontweight="bold", pad=24)
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False, fontsize=7.9)
    axA.grid(axis="y", ls="--", alpha=0.28)
    despine(axA)

    # (B) Delta MCC and Delta Sn (percentage points pp)
    axB = fig.add_subplot(gs[0, 1])
    full_mcc = df_ab.loc["NeuroEGNN (full)", "Full_MCC_mean"]
    full_sn = df_ab.loc["NeuroEGNN (full)", "Full_Sn_mean"] * 100.0
    ab_only = variants[1:]
    y_b = np.arange(len(ab_only))[::-1]
    dmcc = [(df_ab.loc[rk, "Full_MCC_mean"] - full_mcc) * 100.0 for _, rk, _, _, _, _ in ab_only]
    dsn = [(df_ab.loc[rk, "Full_Sn_mean"] * 100.0 - full_sn) for _, rk, _, _, _, _ in ab_only]

    h_b = 0.34
    b_mcc = axB.barh(y_b + h_b / 2, dmcc, height=h_b * 0.90, color=BAR_NAVY, edgecolor="none", lw=0,
                     label="$\\Delta\\mathrm{MCC} \\times 100$")
    b_sn = axB.barh(y_b - h_b / 2, dsn, height=h_b * 0.90, color=BAR_ROSE, edgecolor="none", lw=0,
                    label="$\\Delta\\mathrm{Sensitivity}$ (pp)")
    for yv, vm, vs in zip(y_b, dmcc, dsn):
        axB.text(vm - 0.35, yv + h_b / 2, f"{vm / 100.0:+.4f}", ha="right", va="center", fontsize=7.6, fontweight="bold", color=NAVY)
        axB.text(vs - 0.35, yv - h_b / 2, f"{vs:+.2f} pp", ha="right", va="center", fontsize=7.6, fontweight="bold", color=CRIMSON)
    axB.axvline(0, color="#334155", lw=0.9)
    axB.set_yticks(y_b)
    axB.set_yticklabels(["w/o Coord. Update\n(Static GNN)", "w/o Co-Attention\n(Mean Fusion)", "w/o 3D-EGNN\n(1D ESM-2 Only)", "w/o ESM-2\n(3D-EGNN Only)"], fontsize=7.8)
    axB.set_xlim(-22.2, 1.5)
    axB.set_xlabel("Performance Change Relative to Full NeuroEGNN")
    axB.set_title("(B) Module Removal Impact ($\\Delta\\mathrm{MCC}$ & $\\Delta\\mathrm{Sn}$)", loc="left", fontweight="bold", pad=10)
    axB.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.8)
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
    for v_i, (vk, _, _, v_short, _, v_col) in enumerate(variants):
        means, stds = [], []
        for bk, _ in bins_order:
            sv = df_len[(df_len["Length_Bin"] == bk) & (df_len["Variant"] == vk)]["Sensitivity"] * 100.0
            means.append(sv.mean())
            stds.append(sv.std(ddof=1))
        axC.bar(x_c + (v_i - 2) * w_c, means, w_c * 0.90, yerr=stds, capsize=2.0, error_kw=ERR_KW,
                color=v_col, edgecolor="none", lw=0, label=v_short)
    axC.annotate("+12.28 pp Sn\nvs. w/o 3D-EGNN", xy=(1 - 2 * w_c, 94.2), xytext=(0.45, 98.2),
                 fontsize=7.8, fontweight="bold", color=CRIMSON,
                 arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.1))
    axC.set_ylim(65, 103.5)
    axC.set_xticks(x_c)
    axC.set_xticklabels([bl for _, bl in bins_order], fontsize=8.0)
    axC.set_ylabel("Sensitivity / Recall (%, Mean ± SD)")
    axC.set_title("(C) Length-Stratified Sensitivity Across Regimes", loc="left", fontweight="bold", pad=24)
    axC.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=5, frameon=False, fontsize=7.0, columnspacing=0.9, handletextpad=0.4)
    axC.grid(axis="y", ls="--", alpha=0.28)
    despine(axC)

    # (D) Precision-Sensitivity trade-off across random seeds with iso-F1 contours
    axD = fig.add_subplot(gs[1, 1])
    sn_grid = np.linspace(78, 98, 200)
    pr_grid = np.linspace(77, 97, 200)
    SN_G, PR_G = np.meshgrid(sn_grid, pr_grid)
    F1_G = 2 * (SN_G * PR_G) / (SN_G + PR_G)
    cs = axD.contour(SN_G, PR_G, F1_G, levels=[84, 88, 90, 92, 94], colors="#CBD5E1", linestyles="--", linewidths=0.9)
    axD.clabel(cs, fmt="F1=%.0f%%", fontsize=7.0, colors="#64748B")

    y_te = pd.read_csv(os.path.join(CODE_DIR, "testing_v2.csv"))["Label"].values
    from sklearn.metrics import precision_score, recall_score, f1_score
    markers_d = {"full": "o", "no_coord": "s", "no_attn": "D", "no_3d": "^", "no_esm": "v"}
    for vk, _, _, _, v_long, v_col in variants:
        sns, prs, f1s = [], [], []
        for sd in (500, 501, 502):
            prob = pd.read_csv(os.path.join(CODE_DIR, "results", "ablation", vk, f"seed_{sd}", "test", "test_predictions.csv"))["Prediction_Probability"].values
            pred = (prob >= 0.5).astype(int)
            sns.append(recall_score(y_te, pred) * 100.0)
            prs.append(precision_score(y_te, pred) * 100.0)
            f1s.append(f1_score(y_te, pred) * 100.0)  # per-seed F1, averaged below (matches Table 5)
        axD.scatter(sns, prs, color=v_col, marker=markers_d[vk], s=36, alpha=0.58, edgecolor="white", lw=0.5)
        axD.errorbar(np.mean(sns), np.mean(prs), xerr=np.std(sns, ddof=1), yerr=np.std(prs, ddof=1),
                     fmt=markers_d[vk], color=v_col, markersize=8.5, markeredgecolor="#1E293B", markeredgewidth=0.85,
                     capsize=2.5, lw=1.2, label=f"{v_long} (F1 = {np.mean(f1s):.1f}%)")
    axD.set_xlim(78.5, 97.5)
    axD.set_ylim(78.0, 96.5)
    axD.set_xlabel("Sensitivity / Recall (%)")
    axD.set_ylabel("Precision (%)")
    axD.set_title("(D) Precision–Sensitivity Trade-Off (3 Random Seeds)", loc="left", fontweight="bold", pad=10)
    axD.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.93, fontsize=7.5,
               title="3-seed mean F1", title_fontsize=7.5)
    axD.grid(True, ls=":", alpha=0.25)
    despine(axD)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig3_ablation_and_length.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# Figure 5: fig4_global_interpretability.pdf / .png (5-fold OOF, N=8,887)
def make_fig4_global_interp():
    print("[4/7] Generating fig4_global_interpretability ...")
    fig = plt.figure(figsize=(13.5, 9.8))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.43)

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
    b1 = axA.bar(x_a - w_a / 2, s2g_m, w_a * 0.90, yerr=s2g_s, capsize=3.0, error_kw=ERR_KW,
                 color=BAR_ROSE, edgecolor="none", lw=0,
                 label="1D $\\to$ 3D ($\\mathbf{A}_{\\mathrm{s}\\to\\mathrm{g}}$)")
    b2 = axA.bar(x_a + w_a / 2, g2s_m, w_a * 0.90, yerr=g2s_s, capsize=3.0, error_kw=ERR_KW,
                 color=BAR_BLUE, edgecolor="none", lw=0,
                 label="3D $\\to$ 1D ($\\mathbf{A}_{\\mathrm{g}\\to\\mathrm{s}}$)")
    axA.axhline(1.0, color="#334155", ls="--", lw=1.0, label="Uniform Baseline ($1.0\\times$)")
    for rect, v in zip(b1, s2g_m):
        axA.text(rect.get_x() + rect.get_width() / 2, 0.76, f"{v:.3f}", ha="center", va="center",
                 fontsize=7.8, fontweight="bold", color="white")
    axA.annotate("+25.9% over Local", xy=(2 - w_a / 2, s2g_m[2] + s2g_s[2] + 0.01), xytext=(0.88, 1.40),
                 fontsize=8.2, fontweight="bold", color=CRIMSON,
                 arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.2))
    axA.set_xticks(x_a)
    axA.set_xticklabels(regimes, fontsize=8.0)
    axA.set_ylim(0.65, 1.50)
    axA.set_ylabel("Length-Normalized Cross-Attention ($L \\cdot \\bar{A}_{ij}$)")
    axA.set_title("(A) Cross-Attention Across 3D Contact Regimes", loc="left", fontsize=14.8, fontweight="bold", pad=11)
    axA.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.8)
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
    axB.bar(x_b - w_b / 2, tp_ord, w_b * 0.90, color=BAR_ROSE, edgecolor="none", lw=0, label="True Positives (TP)")
    axB.bar(x_b + w_b / 2, tn_ord, w_b * 0.90, color=BAR_BLUE, edgecolor="none", lw=0, label="True Negatives (TN)")
    axB.axhline(1.0, color="#334155", ls="--", lw=0.9)
    m_pos = aa_ord.index("M")
    axB.annotate("TN $\\mathrm{Met} = 3.74\\times$\n($\\mathrm{Met}_1 = 5.27\\times$)",
                 xy=(m_pos + w_b / 2, tn_ord[m_pos]), xytext=(m_pos + 3.0, 3.25),
                 fontsize=7.9, fontweight="bold", color=NAVY,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#F4F7FA", edgecolor=BAR_BLUE, lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.1))
    axB.set_xticks(x_b)
    axB.set_xticklabels(aa_ord, fontweight="bold", fontsize=8.0)
    axB.set_ylim(0, 4.25)
    axB.set_xlabel("Amino Acid Residue (Sorted by TP Normalized Attention)")
    axB.set_ylabel("Normalized Marginal Attention ($\\times$ Uniform)")
    axB.set_title("(B) Residue-Level Attention Enrichment ($N = 8,887$)", loc="left", fontsize=14.8, fontweight="bold", pad=11)
    axB.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.8)
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
    axC.fill_between(rel_x, tp_rel - 1.96 * tp_sem, tp_rel + 1.96 * tp_sem, color=BAR_ROSE, alpha=0.22)
    axC.plot(rel_x, tn_rel, "-s", color=STEEL, lw=2.0, markersize=4.2, label="True Negatives ($n = 4,035$)")
    axC.fill_between(rel_x, tn_rel - 1.96 * tn_sem, tn_rel + 1.96 * tn_sem, color=BAR_BLUE, alpha=0.22)
    axC.axhline(1.0, color="#334155", ls="--", lw=0.9)
    axC.annotate(f"N-term $\\mathrm{{Met}}_1$ Rejection Spike\nin True Negatives (first bin, ${tn_rel[0]:.2f}\\times$)",
                 xy=(rel_x[0], tn_rel[0]), xytext=(0.18, 3.25),
                 fontsize=7.9, fontweight="bold", color=NAVY,
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.1))
    i_pk = int(np.argmax(tp_rel))
    axC.annotate(f"C-Terminal GPCR Pharmacophore Peak\nin True Positives (bin maximum, ${tp_rel[i_pk]:.2f}\\times$)",
                 xy=(rel_x[i_pk], tp_rel[i_pk]), xytext=(0.30, 2.15),
                 fontsize=7.9, fontweight="bold", color=CRIMSON,
                 arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.1))
    axC.set_xlim(0.0, 1.0)
    axC.set_ylim(0.45, 3.95)
    axC.set_xlabel("Normalized Relative Sequence Position ($0.0 = \\mathrm{N\\text{-}term} \\to 1.0 = \\mathrm{C\\text{-}term}$)")
    axC.set_ylabel("Normalized Attention ($\\times$ Uniform)")
    axC.set_title("(C) Polarized N-to-C Positional Attention", loc="left", fontsize=14.8, fontweight="bold", pad=11)
    axC.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.8)
    axC.grid(True, ls="--", alpha=0.28)
    despine(axC)

    # (D) Out-of-Fold Probability Calibration Histogram (N=8,887)
    axD = fig.add_subplot(gs[1, 1])
    oof_p = OOF_NPZ["oof_NeuroEGNN"]
    oof_y = OOF_NPZ["oof_y"]
    bins_d = np.linspace(0, 1, 26)
    axD.hist(oof_p[oof_y == 0], bins=bins_d, alpha=0.68, color=BAR_BLUE, edgecolor="none", lw=0,
             label="Ground-Truth Negatives ($n = 4,463$)")
    axD.hist(oof_p[oof_y == 1], bins=bins_d, alpha=0.68, color=BAR_ROSE, edgecolor="none", lw=0,
             label="Ground-Truth Neuropeptides ($n = 4,424$)")
    axD.axvline(0.50, color="#334155", ls="--", lw=1.1, label="Decision Boundary ($P = 0.50$)")
    axD.axvspan(0.40, 0.60, color="#F3EFE2", alpha=0.55)
    axD.set_yscale("log")
    axD.set_ylim(3, 4500)
    axD.text(0.50, 900, "77.7% ($6,901 / 8,887$) High-Confidence\n($P \\geq 0.90$ or $P \\leq 0.10$)\nOnly 3.4% ($302$) in $[0.40, 0.60]$",
             ha="center", va="top", fontsize=7.8, fontweight="bold", color="#1E293B",
             bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#CBD5E1", lw=0.9))
    axD.set_xlabel("Out-of-Fold Predicted Neuropeptide Probability $P(\\mathrm{NP})$")
    axD.set_ylabel("Peptide Count (Log Scale)")
    axD.set_title("(D) Out-of-Fold Probability Calibration ($N = 8,887$)", loc="left", fontsize=14.8, fontweight="bold", pad=11)
    axD.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), frameon=True, facecolor="white", framealpha=0.92, fontsize=7.5)
    axD.grid(axis="y", ls="--", alpha=0.28)
    despine(axD)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig4_global_interpretability.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# Figure 6: fig5_case_studies_attention.pdf / .png (hold-out seed_500)
def _soften_cmap(cmap_name, sat_scale=0.52, light_floor=0.18):
    import colorsys
    from matplotlib.colors import LinearSegmentedColormap
    base = plt.get_cmap(cmap_name)
    cols = []
    for t in np.linspace(0.0, 0.92, 256):
        r, g, b, a = base(t)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        s_new = s * sat_scale
        l_new = light_floor + (1.0 - light_floor) * l
        r2, g2, b2 = colorsys.hls_to_rgb(h, l_new, s_new)
        cols.append((r2, g2, b2, a))
    return LinearSegmentedColormap.from_list(f"{cmap_name}_soft", cols)


def make_fig5_cases():
    print("[5/7] Generating fig5_case_studies_attention ...")
    df_test = pd.read_csv(os.path.join(CODE_DIR, "testing_v2.csv"))
    s2g_all = OOF_NPZ["s2g_s500"]
    g2s_all = OOF_NPZ["g2s_s500"]
    probs = OOF_NPZ["ho_egnn_s500"]

    cmap_warm = _soften_cmap("YlOrRd", sat_scale=0.52, light_floor=0.18)
    cmap_cool = _soften_cmap("YlGnBu", sat_scale=0.52, light_floor=0.18)

    fig = plt.figure(figsize=(13.6, 11.6))
    outer = fig.add_gridspec(2, 1, hspace=0.78)

    cases = [
        (608, "(A) True Positive Test-#608 (GIVDQCCTNICSRNQLMSYCND, $L=22$, $P(\\mathrm{NP})=0.9842$ — Disulfide-Bridged Neuropeptide)",
         [5, 6, 10, 19], BAR_TEAL, TEAL, "Cys Disulfide Anchors (1.97× Uniform, 35.8% Share)"),
        (156, "(B) False Negative Test-#156 (MQIYGRPATMDE, $L=12$, $P(\\mathrm{NP})=0.1003 \\to 0.9187$ After Masking $\\mathrm{Met}_1$)",
         [0], BAR_ROSE, CRIMSON, "N-Terminal $\\mathrm{Met}_1$ Suppressor Sink (46.8% $\\mathbf{A}_{\\mathrm{s}\\to\\mathrm{g}}$, 3.57× Uniform)"),
    ]

    for row_idx, (tid, title_str, hi_indices, hi_bar_col, hi_txt_col, hi_lbl) in enumerate(cases):
        seq = str(df_test.iloc[tid]["Seq"]).strip()[:50]
        L = len(seq)
        s2g = s2g_all[tid, :L, :L] * L
        g2s = g2s_all[tid, :L, :L] * L
        s2g_m = s2g.mean(axis=0)
        g2s_m = g2s.mean(axis=0)
        comb = 0.5 * (s2g_m + g2s_m)
        comb_norm = comb / np.mean(comb)

        sub_gs = outer[row_idx].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.25], wspace=0.32)
        ax1 = fig.add_subplot(sub_gs[0, 0])
        ax2 = fig.add_subplot(sub_gs[0, 1])
        ax3 = fig.add_subplot(sub_gs[0, 2])

        res_ticks = [f"{ch}{i+1}" for i, ch in enumerate(seq)]
        vmax = min(5.5, max(np.percentile(s2g, 99), np.percentile(g2s, 99), 2.5))

        im1 = ax1.imshow(s2g, cmap=cmap_warm, vmin=0.0, vmax=vmax, aspect="auto", alpha=0.94)
        ax1.set_title("1D $\\to$ 3D ($\\mathbf{A}_{\\mathrm{s}\\to\\mathrm{g}}$, $\\times$ Uniform)", fontsize=12.2, fontweight="bold", pad=10)
        ax1.set_xlabel("3D Structure Key Residue $j$")
        ax1.set_ylabel("1D Sequence Query Residue $i$")

        im2 = ax2.imshow(g2s, cmap=cmap_cool, vmin=0.0, vmax=vmax, aspect="auto", alpha=0.94)
        ax2.set_title("3D $\\to$ 1D ($\\mathbf{A}_{\\mathrm{g}\\to\\mathrm{s}}$, $\\times$ Uniform)", fontsize=12.2, fontweight="bold", pad=10)
        ax2.set_xlabel("1D Sequence Key Residue $j$")
        ax2.set_ylabel("3D Structure Query Residue $i$")

        step = 1 if L <= 14 else 2
        for ax_h, im in [(ax1, im1), (ax2, im2)]:
            ax_h.set_xticks(np.arange(0, L, step))
            ax_h.set_xticklabels([res_ticks[k] for k in range(0, L, step)],
                                 rotation=45, ha="right", va="top", rotation_mode="anchor", fontsize=7.2)
            ax_h.set_yticks(np.arange(0, L, step))
            ax_h.set_yticklabels([res_ticks[k] for k in range(0, L, step)], fontsize=7.2)
            plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)

        # Marginal bar chart (borderless, desaturated)
        x_r = np.arange(L)
        bar_cols = [hi_bar_col if idx in hi_indices else BAR_MUTED for idx in range(L)]
        bars = ax3.bar(x_r, comb_norm, color=bar_cols, edgecolor="none", lw=0, width=0.68)
        ax3.axhline(1.0, color="#334155", ls="--", lw=0.9, label="Uniform Expectation ($1.0\\times$)")
        for idx in hi_indices:
            dx = -0.24 if (tid == 608 and idx == 5) else (0.24 if (tid == 608 and idx == 6) else 0.0)
            ax3.text(idx + dx, comb_norm[idx] + 0.07, f"{comb_norm[idx]:.2f}×", ha="center", va="bottom",
                     fontsize=7.5, fontweight="bold", color=hi_txt_col)
        # Also label Arg6 on Test-#156 and Arg13 on Test-#608
        extra_idx = 12 if tid == 608 else 5
        ax3.text(extra_idx, comb_norm[extra_idx] + 0.07, f"{comb_norm[extra_idx]:.2f}×", ha="center", va="bottom",
                 fontsize=7.4, fontweight="bold", color=NAVY)
        bars[extra_idx].set_facecolor(BAR_BLUE)

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=hi_bar_col, edgecolor="none", label=hi_lbl),
            Patch(facecolor=BAR_BLUE, edgecolor="none", label=f"Internal Basic Motif ({res_ticks[extra_idx]})"),
        ]
        ax3.set_xticks(x_r)
        ax3.set_xticklabels(res_ticks, rotation=90, ha="center", va="top", fontsize=7.3, fontweight="bold")
        ax3.set_ylim(0, max(comb_norm) * 1.38)
        ax3.set_ylabel("Marginal Attribution ($\\times$ Uniform)")
        ax3.set_title("Combined Residue Attribution Profile", fontsize=12.2, fontweight="bold", pad=10)
        ax3.legend(handles=legend_handles, loc="upper right", fontsize=7.3, frameon=True, facecolor="white", framealpha=0.92)
        ax3.grid(axis="y", ls="--", alpha=0.28)
        despine(ax3)

        ax1.text(-0.18, 1.24, title_str, transform=ax1.transAxes, fontsize=14.6, fontweight="bold", color="#0F172A")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig5_case_studies_attention.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# Figure 7: fig6_structural_relaxation_and_debiased.pdf / .png
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

    fig = plt.figure(figsize=(13.8, 10.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], wspace=0.36, hspace=0.50)

    # (A) Non-Met1 Hard Subset Benchmark (n = 602: 436 Pos / 166 Neg)
    axA = fig.add_subplot(gs[0, 0])
    m_order = [
        ("NeuroPred-CLQ (Bioinf 2022)", "NeuroPred-CLQ\n(2022)"),
        ("UniDL4BioPep (BiB 2023)", "UniDL4BioPep\n(2023)"),
        ("NeuroPred-PLM (BiB 2024)", "NeuroPred-PLM\n(2023)"),
        ("w/o 3D-EGNN (no_3d)", "w/o 3D-EGNN\n(1D ESM-2)"),
        ("w/o Coord Update (no_coord)", "w/o Coord.\n(Static GNN)"),
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

    axA.bar(x_a - w_a, mcc_v, w_a * 0.90, yerr=mcc_e, capsize=2.2, error_kw=ERR_KW,
            color=BAR_NAVY, edgecolor="none", lw=0, label="MCC (×100)")
    axA.bar(x_a, sn_v, w_a * 0.90, yerr=sn_e, capsize=2.2, error_kw=ERR_KW,
            color=BAR_ROSE, edgecolor="none", lw=0, label="Sensitivity (%)")
    axA.bar(x_a + w_a, auc_v, w_a * 0.90, yerr=auc_e, capsize=2.2, error_kw=ERR_KW,
            color=BAR_TEAL, edgecolor="none", lw=0, label="AUC-ROC (×100)")

    axA.annotate("Sn: 95.34% (+8.87 pp vs. w/o 3D-EGNN)\n$\\Delta$MCC = +0.0681 vs. NeuroPred-PLM",
                 xy=(x_a[-1], sn_v[-1] + sn_e[-1] + 0.5), xytext=(1.75, 98.8),
                 fontsize=7.6, fontweight="bold", color=CRIMSON,
                 arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.1))
    axA.set_ylim(50, 105.0)
    axA.set_xticks(x_a)
    axA.set_xticklabels([sl for _, sl in m_order], fontsize=6.8)
    axA.set_ylabel("Non-$\\mathrm{Met}_1$ Subset Score")
    axA.set_title("(A) Debiased Non-$\\mathrm{Met}_1$ Subset ($n = 602$, $\\mathrm{AA}_1 \\neq \\mathrm{M}$)", loc="left", fontsize=14.6, fontweight="bold", pad=32)
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, 1.135), frameon=False, ncol=3, fontsize=7.7)
    axA.grid(axis="y", ls="--", alpha=0.28)
    despine(axA)

    # (B) Rg-normalized E(3)-EGNN coordinate relaxation trajectory
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
    axB2.fill_between(layers, rmsd_q25, rmsd_q75, color=BAR_TEAL, alpha=0.20, label="$\\mathrm{RMSD}_{R_g}$ IQR")
    l3 = axB2.plot(layers, rmsd_m, ":D", color=TEAL, lw=2.0, markersize=5.2,
                   label="Mean $\\mathrm{RMSD}_{R_g}$ ($0.352\\,\\mathrm{\\AA}$)")
    l4 = axB2.plot(layers, rmsd_med, "-.^", color=NAVY, lw=1.6, markersize=4.8,
                   label="Median $\\mathrm{RMSD}_{R_g}$ ($0.038\\,\\mathrm{\\AA}$)")
    axB2.set_ylabel("$R_g$-Normalized Backbone $\\mathrm{RMSD}_{R_g}$ ($\\mathrm{\\AA}$)", color=TEAL)
    axB2.set_ylim(-0.02, 0.45)
    axB2.spines["top"].set_visible(False)

    axB.set_xticks(layers)
    axB.set_xticklabels([f"Layer {l}\n($\\tilde{{\\mathbf{{X}}}}^{{({l})}}$)" for l in layers], fontsize=8.0)
    axB.set_title("(B) $R_g$-Normalized $E(3)$-EGNN Trajectory", loc="left", fontsize=14.6, fontweight="bold", pad=34)
    axB.grid(True, ls="--", alpha=0.28)
    axB.spines["top"].set_visible(False)

    handles_b = l1 + l2 + l3 + l4
    labels_b = [h.get_label() for h in handles_b]
    axB.legend(handles_b, labels_b, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=7.0, frameon=False)

    # (C) Counterfactual Met1 masking on Met1-initiated positive test peptides
    axC = fig.add_subplot(gs[1, 0])
    m1_idx = [i for i, (s, y) in enumerate(zip(bd["raw_seqs"], y_te)) if y == 1 and str(s).strip().startswith("M")]
    p_orig = out_full["probs"][m1_idx]
    p_cf = probs_cf[m1_idx]
    order_c = np.argsort(p_orig)
    y_c = np.arange(len(m1_idx))
    seq_lbls = [f"Test-#{m1_idx[k]} ({bd['raw_seqs'][m1_idx[k]][:9]}..)" for k in order_c]

    axC.axvline(0.50, color=CRIMSON, ls="--", lw=1.2, label="Decision Threshold ($P = 0.50$)")
    for k_pos, ord_k in enumerate(order_c):
        axC.plot([p_orig[ord_k], p_cf[ord_k]], [k_pos, k_pos], color=BAR_TEAL, lw=2.2, zorder=2)
        axC.text(max(p_orig[ord_k], p_cf[ord_k]) + 0.02, k_pos, f"{p_orig[ord_k]:.2f}$\\to${p_cf[ord_k]:.2f}",
                 va="center", ha="left", fontsize=7.3, fontweight="bold", color=TEAL)
    axC.scatter(p_orig[order_c], y_c, color=BAR_AMBER, edgecolor="#6E4730", s=58, zorder=3,
                label="Original (with $\\mathrm{Met}_1$, Recall 16.7%)")
    axC.scatter(p_cf[order_c], y_c, color=BAR_TEAL, edgecolor="#1F4D47", s=66, marker="D", zorder=4,
                label="$\\mathrm{Met}_1$ Masked Out (Rescued, Recall 83.3%)")
    axC.set_yticks(y_c)
    axC.set_yticklabels(seq_lbls, fontsize=7.8)
    axC.set_xlim(-0.03, 1.19)
    axC.set_ylim(-1.65, 5.65)
    axC.set_xlabel("Predicted Neuropeptide Probability $P(\\mathrm{NP})$")
    axC.set_title("(C) Counterfactual Rescue on $\\mathrm{Met}_1$-Positives ($n = 6$)", loc="left", fontsize=14.6, fontweight="bold", pad=11)
    axC.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=7.4)
    axC.grid(axis="x", ls="--", alpha=0.28)
    despine(axC)

    # (D) Latent Manifold Evolution (t-SNE)
    gsD = gs[1, 1].subgridspec(1, 2, wspace=0.14)
    axD1 = fig.add_subplot(gsD[0, 0])
    axD2 = fig.add_subplot(gsD[0, 1])

    tsne_1d = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto").fit_transform(out_no3d["pooled"])
    tsne_full = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto").fit_transform(out_full["pooled"])

    for ax_d, coords, sub_t, k_sep in [
        (axD1, tsne_1d, "w/o 3D-EGNN (1D ESM-2)", "1D ESM-2 Only (no_3d, 256-D)"),
        (axD2, tsne_full, "NeuroEGNN Co-Attn Fused", "NeuroEGNN Co-Attn Fused (256-D)"),
    ]:
        ax_d.scatter(coords[y_te == 0, 0], coords[y_te == 0, 1], c=BAR_BLUE, s=11, alpha=0.58, label="Non-Neuropeptide", rasterized=True)
        ax_d.scatter(coords[y_te == 1, 0], coords[y_te == 1, 1], c=BAR_ROSE, s=11, alpha=0.68, label="Neuropeptide", rasterized=True)
        sil = df_sep.loc[k_sep, "Silhouette_Cosine"]
        fdr = df_sep.loc[k_sep, "Fisher_Discriminant_Ratio"]
        ax_d.set_title(f"{sub_t}\nSil = {sil:.3f} | FDR = {fdr:.2f}", fontsize=10.6, fontweight="bold", pad=7)
        ax_d.set_xticks([])
        ax_d.set_yticks([])
        for sp in ax_d.spines.values():
            sp.set_color("#CBD5E1")
    axD1.text(-0.06, 1.20, "(D) Latent Manifold Separability ($t$-SNE, $N = 888$)",
              transform=axD1.transAxes, fontsize=14.6, fontweight="bold")
    axD2.legend(loc="lower right", markerscale=1.8, fontsize=7.3, frameon=True, facecolor="white", framealpha=0.92)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig6_structural_relaxation_and_debiased.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# Table of Contents (TOC) graphic (3.25 in x 1.75 in)
def make_toc_graphic():
    print("[7/7] Generating fig_toc_graphic (exact 3.25 x 1.75 in) ...")
    import colorsys
    import matplotlib.colors as mcolors
    from matplotlib.patches import Polygon, Arc, Rectangle, Ellipse

    def _mute_rgba(rgba, sat_scale=0.56, light_mix=0.06):
        r, g, b = mcolors.to_rgb(rgba)
        a = rgba[3] if isinstance(rgba, (tuple, list, np.ndarray)) and len(rgba) == 4 else 1.0
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l2 = l + light_mix * (0.95 - l)
        r2, g2, b2 = colorsys.hls_to_rgb(h, l2, s * sat_scale)
        return (r2, g2, b2, a)

    # Exact fig0_architecture scientific palette
    C_SLATE_DARK   = "#253140"
    C_SLATE_TEXT   = "#2D3748"
    C_SLATE_MUTED  = "#5A6677"

    C_NAVY_PRI     = "#2C4D72"
    C_NAVY_DEEP    = "#3D648E"
    C_NAVY_ACC     = "#6083A8"
    C_NAVY_BD      = "#A8BED8"
    C_NAVY_SOFT    = "#E3ECF5"
    C_NAVY_TINT    = "#EFF4FA"

    C_AMBER_DARK   = "#9E6830"
    C_AMBER_MAIN   = "#B8844A"
    C_AMBER_TXT    = "#7C5124"
    C_AMBER_BD     = "#D4B487"
    C_AMBER_STG    = "#DEC7A2"
    C_AMBER_SLAB   = "#EBD6B7"
    C_AMBER_SLABHI = "#ECDBA9"
    C_AMBER_SOFT   = "#F4EBD9"
    C_AMBER_TINT   = "#FAF6EE"
    C_AMBER_BG     = "#FCFBF7"

    C_TEAL_DARK    = "#2B6B65"
    C_TEAL_MAIN    = "#468780"
    C_TEAL_BD      = "#98C4BF"
    C_TEAL_STG     = "#AAD0CB"
    C_TEAL_SOFT    = "#DCECE9"
    C_TEAL_TINT    = "#EEF6F5"
    C_TEAL_BG      = "#F6FAF9"

    C_INDIGO_DARK  = "#46548A"
    C_INDIGO_BD    = "#A6B1D8"
    C_INDIGO_SOFT  = "#E3E7F5"
    C_INDIGO_TINT  = "#EEF1F9"

    C_PURP_DARK    = "#5D468C"
    C_PURP_MAIN    = "#7B65A8"
    C_PURP_BD      = "#C0B4DA"
    C_PURP_SOFT    = "#E8E3F4"
    C_PURP_TINT    = "#F4F2FA"
    C_PURP_BG      = "#F9F8FC"

    C_CRIM_DARK    = "#923F52"
    C_CRIM_MAIN    = "#B25B6E"
    C_CRIM_BD      = "#D9B9C0"
    C_CRIM_SOFT    = "#F1E3E6"
    C_CRIM_TINT    = "#F9F0F2"
    C_CRIM_BG      = "#FBF8F9"

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig = plt.figure(figsize=(3.25, 1.75), dpi=600)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_xlim(0, 325)
        ax.set_ylim(0, 175)
        ax.axis("off")

        ax.add_patch(Rectangle((0, 0), 325, 175, facecolor="white", edgecolor="none", zorder=0))

        box_checks = []  # (text_artist, (xmin, ymin, xmax, ymax), tag)

        def _stage(x, y, w, h, bg, edge, r=4.5, zorder=1):
            ax.add_patch(FancyBboxPatch(
                (x + 1.0, y - 1.0), w, h,
                boxstyle=f"round,pad=0,rounding_size={r}",
                facecolor="#0F172A", edgecolor="none", alpha=0.048, zorder=zorder - 0.3
            ))
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0,rounding_size={r}",
                facecolor=bg, edgecolor=edge, lw=0.82, zorder=zorder
            ))

        def _header(x, y, w, h, title, bg, fontsize=5.3, zorder=4):
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0,rounding_size=2.6",
                facecolor=bg, edgecolor="none", zorder=zorder
            ))
            t = ax.text(x + w / 2, y + h / 2 - 0.15, title,
                        ha="center", va="center", fontsize=fontsize, fontweight="bold",
                        color="white", zorder=zorder + 1)
            box_checks.append((t, (x + 1.0, y + 0.5, x + w - 1.0, y + h - 0.5), f"Header:{title}"))

        def _card(x, y, w, h, title=None, title_bg=None, title_fg=C_SLATE_DARK,
                  bg="white", edge="#CBD5E1", lw=0.65, r=3.2, fontsize=4.5, band_h=9.4, zorder=3):
            ax.add_patch(FancyBboxPatch(
                (x + 0.7, y - 0.7), w, h,
                boxstyle=f"round,pad=0,rounding_size={r}",
                facecolor="#0F172A", edgecolor="none", alpha=0.048, zorder=zorder - 0.2
            ))
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0,rounding_size={r}",
                facecolor=bg, edgecolor=edge, lw=lw, zorder=zorder
            ))
            if title:
                if title_bg:
                    ax.add_patch(FancyBboxPatch(
                        (x + 0.9, y + h - band_h - 0.7), w - 1.8, band_h,
                        boxstyle="round,pad=0,rounding_size=2.1",
                        facecolor=title_bg, edgecolor="none", zorder=zorder + 0.4
                    ))
                t = ax.text(x + w / 2, y + h - band_h / 2 - 0.8, title,
                            ha="center", va="center", fontsize=fontsize, fontweight="bold",
                            color=title_fg, zorder=zorder + 1)
                box_checks.append((t, (x + 1.2, y + h - band_h - 0.7, x + w - 1.2, y + h - 0.5), f"Card:{title}"))

        def _pill(x, y, w, h, text, bg="#F8FAFC", edge="#94A3B8", fg=C_SLATE_TEXT,
                  lw=0.55, fontsize=4.1, r=2.0, zorder=5):
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0,rounding_size={r}",
                facecolor=bg, edgecolor=edge, lw=lw, zorder=zorder
            ))
            t = ax.text(x + w / 2, y + h / 2 - 0.15, text,
                        ha="center", va="center", fontsize=fontsize, fontweight="bold",
                        color=fg, linespacing=1.05, zorder=zorder + 1)
            box_checks.append((t, (x + 0.8, y + 0.4, x + w - 0.8, y + h - 0.4), f"Pill:{text}"))

        def _slab(x, y, w, h, n=3, dx=0.80, dy=0.80, fc="#E7D9A0", ec=C_AMBER_DARK,
                  sub=None, dim="128-D", fg=C_SLATE_DARK, zorder=5):
            for k in range(n - 1, -1, -1):
                al = 0.55 + 0.45 * (1.0 - k / max(1, n - 1))
                ax.add_patch(FancyBboxPatch(
                    (x + k * dx, y + k * dy), w, h,
                    boxstyle="round,pad=0,rounding_size=1.4",
                    facecolor=fc, edgecolor=ec, lw=0.55, alpha=al,
                    zorder=zorder + (n - k) * 0.1
                ))
            if sub:
                # Native bold Helvetica "H" + subscript + "128-D" matching fig0_architecture
                cx = x + w / 2
                yc_top = y + h * 0.70
                yc_bot = y + h * 0.23
                dx_map = {"seq": -2.7, "fuse": -3.3, "graph": -4.9}
                dx_h = dx_map.get(sub, -3.2)
                t_h = ax.text(cx + dx_h, yc_top, "H", ha="center", va="center",
                              fontsize=4.2, fontweight="bold", color=fg, zorder=zorder + 2)
                t_s = ax.text(cx + dx_h + 2.0, yc_top - 0.75, sub, ha="left", va="center",
                              fontsize=3.05, fontweight="bold", color=fg, zorder=zorder + 2)
                t_d = ax.text(cx, yc_bot, dim, ha="center", va="center",
                              fontsize=3.3, fontweight="bold", color=fg, alpha=0.88, zorder=zorder + 2)
                box_checks.append((t_h, (x + 0.6, y + 0.4, x + w - 0.6, y + h - 0.4), f"SlabH:{sub}"))
                box_checks.append((t_s, (x + 0.6, y + 0.4, x + w - 0.6, y + h - 0.4), f"SlabSub:{sub}"))
                box_checks.append((t_d, (x + 0.6, y + 0.4, x + w - 0.6, y + h - 0.4), f"SlabDim:{sub}"))

        def _route(pts, color=C_SLATE_TEXT, lw=1.05, ls="-", style="-|>", ms=5.0, alpha=1.0, zorder=8):
            if len(pts) == 2:
                ax.annotate("", xy=pts[1], xytext=pts[0],
                            arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                            mutation_scale=ms, shrinkA=0, shrinkB=0, alpha=alpha),
                            zorder=zorder)
            else:
                xs = [p[0] for p in pts[:-1]]
                ys = [p[1] for p in pts[:-1]]
                ax.plot(xs, ys, color=color, lw=lw, linestyle=ls, solid_joinstyle="round", alpha=alpha, zorder=zorder)
                ax.annotate("", xy=pts[-1], xytext=pts[-2],
                            arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                            mutation_scale=ms, shrinkA=0, shrinkB=0, alpha=alpha),
                            zorder=zorder)

        def _badge(x, y, text, fg, edge=None, fontsize=3.9, zorder=10):
            ec = edge if edge else fg
            ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=fg,
                    bbox=dict(boxstyle="round,pad=0.16,rounding_size=0.14", facecolor="white",
                              edgecolor=ec, lw=0.55, alpha=0.98),
                    zorder=zorder)

        # STAGE 1 (Left): Homology-Debiased Peptide Input (x = 2.5 .. 92.5)
        _stage(2.5, 2.5, 90.0, 170.0, bg="#F8FAFC", edge="#CBD5E1")
        _header(5.5, 157.4, 84.0, 12.2, "Debiased Input", bg=C_NAVY_PRI, fontsize=5.3)

        # Card 1A (Top): Mature Peptide & Cleavage (y = 93.5 .. 154.2)
        _card(5.5, 93.5, 84.0, 60.7, title="Mature Peptide",
              title_bg="#EEF2F6", title_fg=C_NAVY_PRI, edge="#CBD5E1", fontsize=4.5)

        # Cleaved Met1 box + crimson scissors + 7-bead mature sequence ribbon
        ax.add_patch(FancyBboxPatch(
            (8.5, 122.0), 14.2, 10.2, boxstyle="round,pad=0,rounding_size=1.5",
            facecolor="#E2E8F0", edgecolor="#94A3B8", ls="--", lw=0.58, zorder=4
        ))
        ax.text(15.6, 127.0, "Met₁", ha="center", va="center",
                fontsize=3.9, fontweight="bold", color="#64748B", zorder=5)
        ax.text(15.6, 116.8, "Cut", ha="center", va="center",
                fontsize=3.5, fontweight="bold", color="#64748B", zorder=5)

        ax.plot([25.1, 25.1], [119.2, 138.0], color=C_CRIM_MAIN, lw=0.75, ls=":", zorder=6)
        ax.plot([22.3, 24.5], [137.3, 140.0], color=C_CRIM_MAIN, lw=0.70, zorder=6)
        ax.plot([22.3, 24.5], [140.0, 137.3], color=C_CRIM_MAIN, lw=0.70, zorder=6)
        ax.add_patch(Circle((21.6, 136.9), radius=1.05, facecolor="none", edgecolor=C_CRIM_MAIN, lw=0.55, zorder=6))
        ax.add_patch(Circle((21.6, 140.4), radius=1.05, facecolor="none", edgecolor=C_CRIM_MAIN, lw=0.55, zorder=6))

        beads = [("F", C_CRIM_MAIN), ("A", C_TEAL_MAIN), ("F", C_CRIM_MAIN),
                 ("S", "#64748B"), ("P", C_AMBER_MAIN), ("R", C_NAVY_DEEP), ("L", C_TEAL_MAIN)]
        ax.plot([28.0, 85.0], [127.1, 127.1], color="#94A3B8", lw=1.0, zorder=4)
        for idx_b, (aa_b, col_b) in enumerate(beads):
            bx = 27.6 + idx_b * 8.4
            ax.add_patch(FancyBboxPatch(
                (bx, 122.0), 7.1, 10.2, boxstyle="round,pad=0,rounding_size=1.4",
                facecolor=col_b, edgecolor="white", lw=0.45, zorder=5
            ))
            ax.text(bx + 3.55, 127.0, aa_b, ha="center", va="center",
                    fontsize=4.2, fontweight="bold", color="white", zorder=6)
        ax.plot([61.2, 61.2, 85.1, 85.1], [120.5, 118.8, 118.8, 120.5], color=C_CRIM_MAIN, lw=0.62, zorder=5)
        ax.text(73.1, 115.4, "C-term Core", ha="center", va="center",
                fontsize=3.7, fontweight="bold", color=C_CRIM_DARK, zorder=5)

        _pill(8.5, 97.4, 78.0, 10.2, "Non-Met₁ Debiased",
              bg=C_CRIM_TINT, edge=C_CRIM_BD, fg=C_CRIM_DARK, fontsize=4.1)

        # Card 1B (Bottom): Homology Split (y = 5.5 .. 88.8)
        _card(5.5, 5.5, 84.0, 83.3, title="Homology Split (τ = 0.80)",
              title_bg="#EEF2F6", title_fg=C_NAVY_PRI, edge="#CBD5E1", fontsize=4.3)

        clusters_toc = [
            ([(14.0, 62.5), (23.5, 66.2), (21.8, 53.0), (13.0, 51.5)], C_NAVY_PRI, "Train"),
            ([(43.2, 64.5), (51.5, 57.5), (44.5, 51.5)], C_TEAL_DARK, "Val"),
            ([(70.2, 64.8), (79.2, 57.8), (73.5, 51.5)], C_CRIM_MAIN, "Test"),
        ]
        for c_pts, c_col, c_lbl in clusters_toc:
            cx = np.mean([p[0] for p in c_pts])
            cy = np.mean([p[1] for p in c_pts])
            ax.add_patch(Ellipse((cx, cy), 20.5, 22.0, facecolor=c_col, alpha=0.11,
                                 edgecolor=c_col, ls="--", lw=0.68, zorder=4))
            for i_p in range(len(c_pts)):
                for j_p in range(i_p + 1, len(c_pts)):
                    ax.plot([c_pts[i_p][0], c_pts[j_p][0]], [c_pts[i_p][1], c_pts[j_p][1]],
                            color=c_col, lw=0.75, alpha=0.72, zorder=4)
            for px, py in c_pts:
                ax.add_patch(Circle((px, py), radius=1.9, facecolor=c_col, edgecolor="white", lw=0.5, zorder=5))
            ax.text(cx, 39.2, c_lbl, ha="center", va="center", fontsize=4.0, fontweight="bold",
                    color=c_col, zorder=5)

        for sx_bar in [33.5, 61.0]:
            ax.plot([sx_bar, sx_bar], [36.0, 71.0], color=C_CRIM_MAIN, lw=0.70, ls="--", alpha=0.72, zorder=4)

        _pill(8.5, 22.0, 78.0, 9.8, "7,050 Homology Clusters",
              bg=C_NAVY_TINT, edge=C_NAVY_BD, fg=C_NAVY_PRI, fontsize=4.0)
        _pill(8.5, 9.2, 78.0, 9.8, "Disjoint Splits (Sim < 0.80)",
              bg="#F8FAFC", edge="#CBD5E1", fg=C_SLATE_MUTED, fontsize=4.0)

        # STAGE 2 (Center Hero): NeuroEGNN Dual-Stream Architecture (x = 100.5 .. 226.5)
        _stage(100.5, 2.5, 126.0, 170.0, bg="#F8F8FB", edge="#B8B4CE")
        _header(103.5, 157.4, 120.0, 12.2, "NeuroEGNN Architecture", bg=C_INDIGO_DARK, fontsize=5.3)

        # Module 2A (Top): 1D ESM-2 Evolutionary Stream (y = 113.0 .. 154.2)
        _card(103.5, 113.0, 120.0, 41.2, title="1D ESM-2 (650M)",
              title_bg=C_AMBER_SOFT, title_fg=C_AMBER_DARK, bg=C_AMBER_BG, edge=C_AMBER_BD, fontsize=4.4)

        r_xs = np.linspace(109.2, 169.5, 8)
        ax.plot([r_xs[0], r_xs[-1]], [126.0, 126.0], color=C_AMBER_BD, lw=1.0, zorder=4)
        for rx in r_xs:
            ax.add_patch(Circle((rx, 126.0), radius=1.8, facecolor=C_AMBER_MAIN, edgecolor="white", lw=0.45, zorder=6))
        for ia, ja, ha, lwa in [(0, 3, 9.2, 0.72), (1, 6, 14.0, 0.90), (0, 7, 17.0, 1.05), (4, 7, 9.8, 0.78)]:
            xm = 0.5 * (r_xs[ia] + r_xs[ja])
            wa = r_xs[ja] - r_xs[ia]
            ax.add_patch(Arc((xm, 126.5), wa, ha, angle=0, theta1=0, theta2=180,
                             color=C_AMBER_DARK, lw=lwa, alpha=0.86, zorder=5))
        ax.text(139.3, 118.0, "33 Layers · 20 Heads", ha="center", va="center",
                fontsize=3.7, fontweight="bold", color=C_AMBER_TXT, zorder=5)

        ax.add_patch(Polygon([(175.0, 136.8), (188.6, 132.4), (188.6, 121.6), (175.0, 117.2)],
                             closed=True, facecolor=C_AMBER_SOFT, edgecolor=C_AMBER_MAIN, lw=0.58, zorder=4))
        ax.text(181.2, 127.0, "Proj", ha="center", va="center",
                fontsize=3.6, fontweight="bold", color=C_AMBER_TXT, zorder=5)
        _route([(189.0, 127.0), (195.0, 127.0)], color=C_AMBER_DARK, lw=0.85, ms=4.2)
        _slab(195.8, 117.8, 22.4, 16.2, n=4, dx=0.78, dy=0.78,
              fc=C_AMBER_SLABHI, ec=C_AMBER_DARK, sub="seq", dim="128-D", fg=C_AMBER_TXT)

        # Module 2B (Middle Bridge, inset right of vertical C lane): Gated Co-Attention (x = 116.5 .. 223.5, y = 63.5 .. 103.8)
        _card(116.5, 63.5, 107.0, 40.3, title="Gated Co-Attention (4H)",
              title_bg=C_PURP_SOFT, title_fg=C_PURP_DARK, bg=C_PURP_BG, edge=C_PURP_BD, fontsize=4.4)

        # Dual 6x6 Cross-Attention Matrix (Purples upper s->g, Blues lower g->s)
        ax0_s, ay0_s, asz = 120.5, 68.2, 19.8
        cs_a = asz / 6.0
        ca_mat = np.array([
            [0.45, 0.20, 0.85, 0.22, 0.90, 0.30],
            [0.35, 0.55, 0.78, 0.20, 0.82, 0.25],
            [0.25, 0.30, 0.92, 0.28, 0.88, 0.35],
            [0.20, 0.28, 0.35, 0.65, 0.80, 0.75],
            [0.25, 0.32, 0.76, 0.35, 0.92, 0.84],
            [0.22, 0.25, 0.38, 0.68, 0.82, 0.88],
        ])
        for ri in range(6):
            for ci in range(6):
                cmap_fn = plt.cm.Purples if ci >= ri else plt.cm.Blues
                ax.add_patch(Rectangle(
                    (ax0_s + ci * cs_a, ay0_s + (5 - ri) * cs_a), cs_a, cs_a,
                    facecolor=_mute_rgba(cmap_fn(ca_mat[ri, ci])), edgecolor="none", zorder=4
                ))
        ax.plot([ax0_s, ax0_s + asz], [ay0_s + asz, ay0_s], color="white", lw=0.55, alpha=0.85, zorder=4.5)
        ax.add_patch(Rectangle((ax0_s, ay0_s), asz, asz, facecolor="none", edgecolor=C_PURP_DARK, lw=0.60, zorder=5))

        _route([(141.0, 78.1), (146.4, 78.1)], color=C_PURP_DARK, lw=0.82, ms=4.0)

        # Sigmoid Gate Pill + Vector Plus Residual Fusion Node
        ax.add_patch(FancyBboxPatch(
            (146.8, 71.8), 28.8, 12.6, boxstyle="round,pad=0,rounding_size=2.0",
            facecolor=C_INDIGO_TINT, edgecolor=C_INDIGO_DARK, lw=0.60, zorder=5
        ))
        st = np.linspace(-3, 3, 25)
        sx_g = 152.8 + 3.8 * (st / 3.0)
        sy_g = 78.1 + 3.6 * (1.0 / (1.0 + np.exp(-st)) - 0.5) * 2.0
        ax.plot(sx_g, sy_g, color=C_CRIM_MAIN, lw=1.10, zorder=6)
        ax.text(166.6, 78.0, "Gate", ha="center", va="center",
                fontsize=4.0, fontweight="bold", color=C_INDIGO_DARK, zorder=6)

        _route([(176.0, 78.1), (180.8, 78.1)], color=C_INDIGO_DARK, lw=0.82, ms=4.0)
        ax.add_patch(Circle((184.8, 78.1), radius=3.5, facecolor=C_INDIGO_DARK, edgecolor="white", lw=0.5, zorder=6))
        ax.plot([183.1, 186.5], [78.1, 78.1], color="white", lw=0.95, solid_capstyle="round", zorder=7)
        ax.plot([184.8, 184.8], [76.4, 79.8], color="white", lw=0.95, solid_capstyle="round", zorder=7)

        _route([(188.6, 78.1), (195.6, 78.1)], color=C_INDIGO_DARK, lw=0.85, ms=4.2)
        _slab(196.4, 70.3, 21.2, 14.8, n=3, dx=0.82, dy=0.82,
              fc=C_INDIGO_SOFT, ec=C_INDIGO_DARK, sub="fuse", dim="128-D", fg=C_INDIGO_DARK)

        # Module 2C (Bottom): 4-Layer E(3)-Equivariant GNN (y = 5.5 .. 54.8)
        _card(103.5, 5.5, 120.0, 49.3, title="4-Layer E(3)-EGNN",
              title_bg=C_TEAL_SOFT, title_fg=C_TEAL_DARK, bg=C_TEAL_BG, edge=C_TEAL_MAIN, fontsize=4.4)

        # Contact map C (7x7)
        ig = np.arange(7)
        dg = np.abs(ig[:, None] - ig[None, :])
        cm_c = np.exp(-dg / 1.8)
        cm_c[1, 5] = cm_c[5, 1] = 0.88
        cm_c[0, 6] = cm_c[6, 0] = 0.80
        mx0, my0, msz = 106.8, 19.5, 18.2
        cs_m = msz / 7.0
        for ri in range(7):
            for ci in range(7):
                ax.add_patch(Rectangle(
                    (mx0 + ci * cs_m, my0 + (6 - ri) * cs_m), cs_m, cs_m,
                    facecolor=_mute_rgba(plt.cm.YlGnBu(cm_c[ri, ci] * 0.85 + 0.10)), edgecolor="none", zorder=4
                ))
        ax.add_patch(Rectangle((mx0, my0), msz, msz, facecolor="none", edgecolor=C_NAVY_PRI, lw=0.58, zorder=5))
        ax.text(mx0 + msz / 2, 11.8, "Map C", ha="center", va="center",
                fontsize=3.7, fontweight="bold", color=C_NAVY_PRI, zorder=5)

        _route([(125.6, 28.6), (131.6, 28.6)], color=C_TEAL_DARK, lw=0.80, ms=3.8)

        # 3D C-alpha folded peptide
        p3d = [(133.6, 35.8), (138.9, 38.6), (145.6, 37.0), (151.4, 32.2),
               (152.4, 24.8), (147.2, 19.8), (139.4, 19.5), (134.6, 25.0)]
        for u3, v3 in [(0, 7), (1, 6), (2, 5)]:
            ax.plot([p3d[u3][0], p3d[v3][0]], [p3d[u3][1], p3d[v3][1]],
                    color=C_CRIM_MAIN, ls="--", lw=0.68, alpha=0.85, zorder=4)
        ax.plot([p[0] for p in p3d], [p[1] for p in p3d], color=C_TEAL_DARK, lw=1.25, zorder=5)
        for ip, (px, py) in enumerate(p3d):
            ncol = C_CRIM_MAIN if ip in (2, 5, 7) else C_TEAL_MAIN
            ax.add_patch(Circle((px, py), radius=1.6, facecolor=ncol, edgecolor="white", lw=0.42, zorder=6))
        ax.text(143.2, 11.8, "3D < 10 Å", ha="center", va="center",
                fontsize=3.7, fontweight="bold", color=C_TEAL_DARK, zorder=5)

        _route([(154.4, 28.6), (160.4, 28.6)], color=C_TEAL_DARK, lw=0.80, ms=3.8)

        # Equivariant Radial Force Node + E(3) Relaxation
        cix, ciy = 171.6, 29.0
        ax.add_patch(Circle((cix, ciy), radius=9.0, facecolor=C_TEAL_TINT, edgecolor=C_TEAL_MAIN,
                            ls=":", lw=0.58, alpha=0.72, zorder=4))
        for nx, ny in [(163.3, 35.6), (163.8, 22.2), (180.4, 36.0)]:
            ax.plot([cix, nx], [ciy, ny], color="#94A3B8", lw=0.65, zorder=4.5)
            fx = cix + 0.56 * (nx - cix)
            fy = ciy + 0.56 * (ny - ciy)
            _route([(cix, ciy), (fx, fy)], color=C_CRIM_MAIN, lw=0.85, style="->", ms=4.0, zorder=6)
            ax.add_patch(Circle((nx, ny), radius=1.8, facecolor=C_NAVY_PRI, edgecolor="white", lw=0.4, zorder=6))
        ax.add_patch(Circle((cix, ciy), radius=2.5, facecolor=C_TEAL_DARK, edgecolor="white", lw=0.5, zorder=7))
        ax.text(cix, ciy - 0.1, "i", ha="center", va="center", fontsize=3.7, fontweight="bold",
                style="italic", color="white", zorder=8)
        _route([(cix, ciy), (179.8, 22.6)], color=C_AMBER_DARK, lw=0.90, ls="--", style="->", ms=4.4, zorder=7)
        ax.add_patch(Circle((179.8, 22.6), radius=1.9, facecolor=C_AMBER_MAIN, edgecolor="white", lw=0.4, zorder=7))
        ax.text(172.2, 11.8, "−229 mÅ", ha="center", va="center",
                fontsize=3.7, fontweight="bold", color=C_CRIM_DARK, zorder=5)

        _route([(183.0, 28.6), (195.6, 28.6)], color=C_TEAL_DARK, lw=0.85, ms=4.2)
        _slab(196.4, 19.6, 21.2, 15.2, n=3, dx=0.82, dy=0.82,
              fc=C_TEAL_SOFT, ec=C_TEAL_DARK, sub="graph", dim="128-D", fg=C_TEAL_DARK)

        # Internal Stage 2 Vertical Arrows: Dedicated C Lane + Co-Attention Q/K/V & Residual Fusion
        _route([(109.8, 113.0), (109.8, 54.8)], color=C_AMBER_DARK, lw=0.95, ls="--", ms=4.5, zorder=7)
        _badge(109.8, 83.6, "C", fg=C_AMBER_DARK, edge=C_AMBER_BD, fontsize=4.0)

        _route([(130.4, 113.0), (130.4, 103.8)], color=C_AMBER_DARK, lw=1.0, ms=4.5)
        _route([(184.8, 113.0), (184.8, 103.8)], color=C_AMBER_DARK, lw=1.0, ms=4.5)
        _route([(130.4, 54.8), (130.4, 63.5)], color=C_TEAL_DARK, lw=1.0, ms=4.5)
        _route([(184.8, 54.8), (184.8, 63.5)], color=C_TEAL_DARK, lw=1.0, ms=4.5)

        # STAGE 3 (Right): GPCR Pharmacophore & SOTA Benchmark (x = 234.5 .. 322.5)
        _stage(234.5, 2.5, 88.0, 170.0, bg=C_CRIM_BG, edge=C_CRIM_BD)
        _header(237.5, 157.4, 82.0, 12.2, "Motif & Benchmark", bg=C_CRIM_DARK, fontsize=5.2)

        # Card 3A (Top): GPCR Core Motif Attribution (y = 95.0 .. 154.2)
        _card(237.5, 95.0, 82.0, 59.2, title="GPCR Core Motif",
              title_bg=C_CRIM_SOFT, title_fg=C_CRIM_DARK, edge=C_CRIM_BD, fontsize=4.5)

        seq_a = ["S", "G", "G", "F", "A", "F", "S", "P", "R", "L"]
        h_att = np.array([0.22, 0.18, 0.24, 0.72, 0.30, 0.68, 0.28, 0.84, 0.98, 0.90]) * 24.5
        xs_att = np.linspace(243.2, 313.8, 10)
        # Uniform 1.0x reference baseline across motif profile
        ax.plot([240.0, 317.0], [114.2, 114.2], color="#94A3B8", lw=0.55, ls="--", zorder=4)
        ax.text(251.0, 118.2, "1.0× Avg", ha="center", va="center",
                fontsize=3.5, fontweight="bold", color="#64748B", zorder=6)
        for xp_a, hp_a, ch_a in zip(xs_att, h_att, seq_a):
            bc_a = C_CRIM_MAIN if hp_a > 12.0 else "#94A3B8"
            ax.add_patch(Rectangle((xp_a - 2.4, 105.5), 4.8, hp_a, facecolor=bc_a, edgecolor="none", alpha=0.92, zorder=5))
            ax.text(xp_a, 100.2, ch_a, ha="center", va="center", fontsize=4.0, fontweight="bold", color=C_SLATE_TEXT, zorder=6)
        ax.plot([240.0, 317.0], [105.5, 105.5], color="#64748B", lw=0.60, zorder=5)
        ax.plot([295.5, 295.5, 316.2, 316.2], [131.5, 133.4, 133.4, 131.5], color=C_CRIM_MAIN, lw=0.62, zorder=5)
        ax.text(294.0, 137.5, "1.93× C-term", ha="center", va="center",
                fontsize=3.9, fontweight="bold", color=C_CRIM_DARK, zorder=6)

        # Card 3B (Bottom): 5-Fold CV MCC Benchmark (y = 5.5 .. 90.5)
        _card(237.5, 5.5, 82.0, 85.0, title="5-Fold CV MCC",
              title_bg="#EEF2F6", title_fg=C_NAVY_PRI, edge="#CBD5E1", fontsize=4.6)

        mcc_rows = [
            ("NeuroEGNN",     0.8421, ".842", C_CRIM_MAIN),
            ("UniDL4BioPep",  0.8156, ".816", C_NAVY_DEEP),
            ("NeuroPred-PLM", 0.7982, ".798", C_TEAL_MAIN),
            ("PredNeuroP",    0.7820, ".782", "#8492A6"),
        ]
        ys_bar = [70.2, 57.8, 45.4, 33.0]
        bx0 = 240.6
        max_bw = 57.2
        for (m_lbl, m_val, m_str, m_col), by in zip(mcc_rows, ys_bar):
            bw = max_bw * ((m_val - 0.64) / (0.8421 - 0.64))
            ax.add_patch(FancyBboxPatch(
                (bx0, by - 4.6), bw, 9.2, boxstyle="round,pad=0,rounding_size=1.8",
                facecolor=m_col, edgecolor="none", zorder=5
            ))
            t_in = ax.text(bx0 + 1.9, by - 0.1, m_lbl, ha="left", va="center",
                           fontsize=3.8, fontweight="bold", color="white", zorder=6)
            box_checks.append((t_in, (bx0 + 0.4, by - 4.6, bx0 + bw - 0.4, by + 4.6), f"Bar:{m_lbl}"))
            ax.text(bx0 + bw + 1.6, by - 0.1, m_str, ha="left", va="center",
                    fontsize=4.15, fontweight="bold",
                    color=C_CRIM_DARK if m_lbl == "NeuroEGNN" else C_SLATE_TEXT, zorder=6)

        _pill(241.0, 9.2, 36.8, 11.2, "AUC 0.978",
              bg=C_CRIM_TINT, edge=C_CRIM_BD, fg=C_CRIM_DARK, fontsize=4.1)
        _pill(279.7, 9.2, 36.8, 11.2, "Sn 93.7%",
              bg=C_TEAL_TINT, edge=C_TEAL_BD, fg=C_TEAL_DARK, fontsize=4.1)

        # INTER-STAGE ROUTED ARROWS (Stage 1 -> Stage 2 -> Stage 3)
        # Stage 1 -> Stage 2 Clean T-Junction (Sequence -> ESM-2 & Peptide -> 3D EGNN)
        ax.plot([89.5, 96.2], [127.1, 127.1], color=C_NAVY_PRI, lw=1.10, zorder=8)
        _route([(96.2, 127.1), (103.5, 127.1)], color=C_AMBER_DARK, lw=1.10, ms=4.8)
        _route([(96.2, 127.1), (96.2, 28.6), (103.5, 28.6)], color=C_TEAL_DARK, lw=1.05, ms=4.8)
        ax.add_patch(Circle((96.2, 127.1), radius=1.25, facecolor=C_NAVY_PRI, edgecolor="white", lw=0.40, zorder=9))

        # Stage 2 -> Stage 3 Clean T-Junction (H_fuse -> GPCR Motif & SOTA Classification)
        ax.plot([219.5, 230.5], [78.1, 78.1], color=C_INDIGO_DARK, lw=1.10, zorder=8)
        _route([(230.5, 78.1), (230.5, 124.0), (237.5, 124.0)], color=C_INDIGO_DARK, lw=1.10, ms=4.8)
        _route([(230.5, 78.1), (230.5, 52.0), (237.5, 52.0)], color=C_CRIM_DARK, lw=1.05, ms=4.8)
        ax.add_patch(Circle((230.5, 78.1), radius=1.25, facecolor=C_INDIGO_DARK, edgecolor="white", lw=0.40, zorder=9))

        # Programmatic BBox Overflow & Overlap Audit (filtering empty annotation placeholders)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        n_overflow = 0
        for t_obj, (bx1, by1, bx2, by2), tag in box_checks:
            bb = inv.transform_bbox(t_obj.get_window_extent(renderer=renderer))
            if bb.x0 < bx1 - 0.8 or bb.x1 > bx2 + 0.8 or bb.y0 < by1 - 0.8 or bb.y1 > by2 + 0.8:
                print(f"  [WARN OVERFLOW] {tag}: text=({bb.x0:.1f},{bb.y0:.1f},{bb.x1:.1f},{bb.y1:.1f}) vs box=({bx1:.1f},{by1:.1f},{bx2:.1f},{by2:.1f})")
                n_overflow += 1
        all_t = [(t, inv.transform_bbox(t.get_window_extent(renderer=renderer)))
                 for t in ax.texts if t.get_text().strip()]
        n_overlap = 0
        for i_t in range(len(all_t)):
            for j_t in range(i_t + 1, len(all_t)):
                b1, b2 = all_t[i_t][1], all_t[j_t][1]
                if (b1.x0 < b2.x1 - 0.35 and b1.x1 > b2.x0 + 0.35 and
                        b1.y0 < b2.y1 - 0.35 and b1.y1 > b2.y0 + 0.35):
                    print(f"  [WARN OVERLAP] '{all_t[i_t][0].get_text()}' <-> '{all_t[j_t][0].get_text()}'")
                    n_overlap += 1
        print(f"  [TOC BBox Audit] Overflows: {n_overflow} | Text-Text Overlaps: {n_overlap}")

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
