# -*- coding: utf-8 -*-
"""Assemble Figure 7 (fig5b_3d_structural_motifs.pdf/.png) from PyMOL-rendered 3D conformers."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_PAPER_FIG = os.path.join(os.path.dirname(CODE_DIR), "paper", "figures")
FIG_DIR = _PAPER_FIG if os.path.isdir(_PAPER_FIG) else os.path.join(CODE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
PYMOL_DIR = os.path.join(CODE_DIR, "results", "pymol_sessions")
DATA_PATH = os.path.join(CODE_DIR, "results", "figure_data", "case_3d_data.npz")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.8,
    "axes.titlesize": 16.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})

NAVY = "#3D556E"
BURGUNDY = "#914153"
TEAL = "#4D7A75"
GOLD_DARK = "#8F6518"
SLATE = "#64748B"
DARK_INK = "#1E293B"


def make_attr_cmap():
    stops = [
        (0.00, "#CBD5E1"),
        (0.26, "#8EA5BC"),
        (0.54, "#D7B98E"),
        (0.80, "#C07A68"),
        (1.00, "#914153"),
    ]
    return mcolors.LinearSegmentedColormap.from_list("neuro_attr_3d", stops, N=256)


CMAP_ATTR = make_attr_cmap()
NORM_ATTR = mcolors.Normalize(vmin=0.45, vmax=2.75)


def compute_comb_norm(s2g, g2s):
    L = s2g.shape[0]
    comb = 0.5 * ((s2g * L).mean(axis=0) + (g2s * L).mean(axis=0))
    return comb / np.mean(comb)


def load_panel_image(ax, png_name, meta_entry):
    img = mpimg.imread(os.path.join(PYMOL_DIR, png_name))
    w = meta_entry["width"]
    h = meta_entry["height"]
    ratio = w / float(h)
    ax.imshow(img, extent=[0.0, ratio, 0.0, 1.0], origin="upper", interpolation="lanczos", zorder=2)
    ax.set_xlim(0.0, ratio)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#CBD5E1")
        sp.set_linewidth(0.95)
    return ratio


def pt_xy(meta_entry, key):
    ratio = meta_entry["width"] / float(meta_entry["height"])
    u, v = meta_entry["pts"][key]
    return (u * ratio, v)


def draw_callouts(ax, items, x_anchor, side="left", fontsize=7.5):
    ha = "left" if side == "left" else "right"
    items_sorted = sorted(items, key=lambda it: it["pt"][1], reverse=True)
    slots_sorted = sorted([it["y"] for it in items], reverse=True)
    for it, ty in zip(items_sorted, slots_sorted):
        px, py = it["pt"]
        col = it.get("color", DARK_INK)
        bcol = it.get("border", "#CBD5E1")
        bg = it.get("bg", "#FFFFFF")
        ax.scatter([px], [py], color=col, s=11, zorder=15)
        ax.annotate(
            it["text"],
            xy=(px, py),
            xytext=(x_anchor, ty),
            ha=ha, va="center",
            fontsize=fontsize, fontweight="bold", color=col,
            bbox=dict(boxstyle="round,pad=0.23", facecolor=bg, edgecolor=bcol, lw=0.85, alpha=0.96),
            arrowprops=dict(arrowstyle="-", color=col, lw=0.95, alpha=0.84, shrinkA=0, shrinkB=2),
            zorder=16,
        )


def main():
    d = np.load(DATA_PATH, allow_pickle=True)
    with open(os.path.join(PYMOL_DIR, "pymol_views.json")) as f:
        meta = json.load(f)

    attr608 = compute_comb_norm(d["s2g_608"], d["g2s_608"])
    attr216 = compute_comb_norm(d["s2g_216"], d["g2s_216"])
    attr651 = compute_comb_norm(d["s2g_651"], d["g2s_651"])
    attr156_b = compute_comb_norm(d["s2g_156"], d["g2s_156"])
    attr156_a = np.concatenate([[0.45], compute_comb_norm(d["s2g_cf_156"][1:, 1:], d["g2s_cf_156"][1:, 1:])])

    fig = plt.figure(figsize=(14.4, 11.0))
    gs = fig.add_gridspec(2, 2, wspace=0.10, hspace=0.24, bottom=0.085, top=0.935, left=0.025, right=0.975)

    # Panel A: Test-#608
    axA = fig.add_subplot(gs[0, 0])
    rA = load_panel_image(axA, "panelA_test608.png", meta["A"])
    mA = meta["A"]

    cands_A = [
        ("CA_1",   "N-term Gly1", SLATE, "#CBD5E1", "#FFFFFF"),
        ("TIP_6",  f"Cys6 ({attr608[5]:.2f}×)", BURGUNDY, "#D8A7B1", "#FFF7F8"),
        ("TIP_7",  f"Cys7 ({attr608[6]:.2f}×)", BURGUNDY, "#D8A7B1", "#FFF7F8"),
        ("TIP_11", f"Cys11 ({attr608[10]:.2f}×, S–S 1.94 Å)", GOLD_DARK, "#E2C37E", "#FFFDF5"),
        ("TIP_13", f"Arg13 ({attr608[12]:.2f}×)", NAVY, "#94A3B8", "#F8FAFC"),
        ("TIP_20", f"Cys20 ({attr608[19]:.2f}×)", BURGUNDY, "#D8A7B1", "#FFF7F8"),
    ]
    left_A, right_A = [], []
    for k_pt, txt, col, bcol, bg in cands_A:
        p = pt_xy(mA, k_pt)
        (left_A if p[0] < 0.5 * rA else right_A).append(dict(pt=p, text=txt, color=col, border=bcol, bg=bg))
    for idx_s, it in enumerate(left_A):
        it["y"] = 0.72 - idx_s * 0.22
    for idx_s, it in enumerate(right_A):
        it["y"] = 0.74 - idx_s * 0.19
    draw_callouts(axA, left_A, 0.035, side="left")
    draw_callouts(axA, right_A, rA - 0.035, side="right")

    axA.text(0.025, 0.968,
             "Test-#608 (Insulin-like Two-Helix Fold, $L=22$ aa)  |  $P(\\mathrm{NP}) = 0.9842$\n"
             "Covalent $\\mathrm{Cys}_6$–$\\mathrm{Cys}_{11}$ Disulfide ($d_{\\mathrm{S\\gamma-S\\gamma}} = 1.94$ Å)  |  4 Cys Share = $35.8\\%$ ($1.97\\times$)",
             transform=axA.transAxes, ha="left", va="top", fontsize=7.4, fontweight="bold", color=DARK_INK,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.95), zorder=20)
    axA.set_title("(A) 3D Disulfide-Stabilized Helical Scaffold", loc="left", fontsize=16.4, fontweight="bold", pad=10)

    # Panel B: Test-#216
    axB = fig.add_subplot(gs[0, 1])
    rB = load_panel_image(axB, "panelB_test216.png", meta["B"])
    mB = meta["B"]

    cands_B = [
        ("CA_1",   f"N-term Asp1 ({attr216[0]:.2f}×)", SLATE, "#CBD5E1", "#FFFFFF"),
        ("TIP_6",  "Phe6–Phe7 Aromatic Helix", TEAL, "#8BB8B2", "#F4FBF9"),
        ("TIP_12", f"Lys12 ({attr216[11]:.2f}×)", NAVY, "#94A3B8", "#F8FAFC"),
        ("CA_17",  f"Pro17 Turn ({attr216[16]:.2f}×)", NAVY, "#94A3B8", "#F8FAFC"),
        ("TIP_18", f"Arg18 Anchor ({attr216[17]:.2f}×)", BURGUNDY, "#D8A7B1", "#FFF7F8"),
        ("CA_19",  f"C-term Val19 ({attr216[18]:.2f}×)", SLATE, "#CBD5E1", "#FFFFFF"),
    ]
    left_B, right_B = [], []
    for k_pt, txt, col, bcol, bg in cands_B:
        p = pt_xy(mB, k_pt)
        (left_B if p[0] < 0.52 * rB else right_B).append(dict(pt=p, text=txt, color=col, border=bcol, bg=bg))
    for idx_s, it in enumerate(left_B):
        it["y"] = 0.72 - idx_s * 0.24
    for idx_s, it in enumerate(right_B):
        it["y"] = 0.72 - idx_s * 0.24
    draw_callouts(axB, left_B, 0.035, side="left")
    draw_callouts(axB, right_B, rB - 0.035, side="right")

    cterm216 = attr216[-3:].mean() / attr216[:-3].mean()
    axB.text(0.025, 0.968,
             f"Test-#216 (DEVPAFFLKIAKMKTLPRV, $L=19$ aa)  |  1D $P = 0.3602 \\to$ NeuroEGNN $P = 0.9433$\n"
             f"Amphipathic $\\alpha$-Helix + C-Terminal -PRV Turn Polarity = ${cterm216:.2f}\\times$ Core",
             transform=axB.transAxes, ha="left", va="top", fontsize=7.35, fontweight="bold", color=DARK_INK,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.95), zorder=20)
    axB.set_title("(B) Amphipathic Helix & 3D Conformational Rescue", loc="left", fontsize=16.4, fontweight="bold", pad=10)

    # Panel C: Test-#651
    axC = fig.add_subplot(gs[1, 0])
    rC = load_panel_image(axC, "panelC_test651.png", meta["C"])
    mC = meta["C"]
    cterm651 = attr651[-3:].mean() / attr651[:-3].mean()

    cands_C = [
        ("CA_1",   f"N-term Disordered Tail ({attr651[:6].mean():.2f}×)", SLATE, "#CBD5E1", "#FFFFFF"),
        ("TIP_14", f"Phe14 Aromatic Ring ({attr651[13]:.2f}×)", TEAL, "#8BB8B2", "#F4FBF9"),
        ("CA_16",  f"Pro16 β-Turn Kink ({attr651[15]:.2f}×)", NAVY, "#94A3B8", "#F8FAFC"),
        ("TIP_17", f"Arg17 GPCR Anchor ({attr651[16]:.2f}×)", BURGUNDY, "#D8A7B1", "#FFF7F8"),
        ("CA_18",  f"C-term Leu18 ({attr651[17]:.2f}×)", BURGUNDY, "#D8A7B1", "#FFF7F8"),
    ]
    left_C, right_C = [], []
    for k_pt, txt, col, bcol, bg in cands_C:
        p = pt_xy(mC, k_pt)
        (left_C if p[0] < 0.48 * rC else right_C).append(dict(pt=p, text=txt, color=col, border=bcol, bg=bg))
    for idx_s, it in enumerate(left_C):
        it["y"] = 0.76 - idx_s * 0.17
    for idx_s, it in enumerate(right_C):
        it["y"] = 0.68 - idx_s * 0.22
    draw_callouts(axC, left_C, 0.035, side="left")
    draw_callouts(axC, right_C, rC - 0.035, side="right")

    axC.text(0.025, 0.968,
             f"Test-#651 (Pyrokinin DASSSNENNSRPPFAPRL, $L=18$ aa)  |  $P(\\mathrm{{NP}}) = 0.9918$\n"
             f"Conserved C-Terminal -PPFAPRL $\\beta$-Turn Pharmacophore Enrichment = ${cterm651:.2f}\\times$ Core",
             transform=axC.transAxes, ha="left", va="top", fontsize=7.4, fontweight="bold", color=DARK_INK,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.95), zorder=20)
    axC.set_title("(C) Conserved C-Terminal -FXPRL $\\beta$-Turn Pocket", loc="left", fontsize=16.4, fontweight="bold", pad=10)

    # Panel D: Test-#156 before and after Met1 masking
    subD = gs[1, 1].subgridspec(1, 2, wspace=0.04)
    axD1 = fig.add_subplot(subD[0, 0])
    axD2 = fig.add_subplot(subD[0, 1])

    rD1 = load_panel_image(axD1, "D_before_test156.png", meta["D_before"])
    mD1 = meta["D_before"]
    draw_callouts(axD1, [
        dict(pt=pt_xy(mD1, "CA_1"),  y=0.72, text=f"Met1 Sink ({attr156_b[0]:.2f}×)", color=BURGUNDY, border="#D8A7B1", bg="#FFF7F8"),
        dict(pt=pt_xy(mD1, "TIP_4"), y=0.14, text=f"Tyr4 ({attr156_b[3]:.2f}×)", color=SLATE, border="#CBD5E1", bg="#FFFFFF"),
    ], 0.025, side="left", fontsize=7.0)
    draw_callouts(axD1, [
        dict(pt=pt_xy(mD1, "TIP_6"),  y=0.78, text=f"Arg6 ({attr156_b[5]:.2f}×)", color=NAVY, border="#94A3B8", bg="#F8FAFC"),
        dict(pt=pt_xy(mD1, "TIP_10"), y=0.14, text=f"Met10 ({attr156_b[9]:.2f}×)", color=SLATE, border="#CBD5E1", bg="#FFFFFF"),
    ], rD1 - 0.025, side="right", fontsize=7.0)
    axD1.text(0.50, 0.972, "Unmasked ($\\mathrm{Met}_1$ Present)\n$P(\\mathrm{NP}) = 0.1003$ (FN, 46.8% on M1)",
              transform=axD1.transAxes, ha="center", va="top", fontsize=7.0, fontweight="bold", color=BURGUNDY,
              bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF7F8", edgecolor="#D8A7B1", alpha=0.96), zorder=20)

    rD2 = load_panel_image(axD2, "D_after_test156.png", meta["D_after"])
    mD2 = meta["D_after"]
    draw_callouts(axD2, [
        dict(pt=pt_xy(mD2, "CA_1"),  y=0.78, text="Met1 Masked ($M_1=0$)", color=SLATE, border="#CBD5E1", bg="#F8FAFC"),
        dict(pt=pt_xy(mD2, "TIP_6"), y=0.52, text=f"Arg6 ({attr156_a[5]:.2f}×)", color=BURGUNDY, border="#D8A7B1", bg="#FFF7F8"),
        dict(pt=pt_xy(mD2, "TIP_4"), y=0.14, text=f"Tyr4 ({attr156_a[3]:.2f}×)", color=TEAL, border="#8BB8B2", bg="#F4FBF9"),
    ], 0.025, side="left", fontsize=7.0)
    draw_callouts(axD2, [
        dict(pt=pt_xy(mD2, "CA_9"),   y=0.78, text=f"Thr9 ({attr156_a[8]:.2f}×)", color=TEAL, border="#8BB8B2", bg="#F4FBF9"),
        dict(pt=pt_xy(mD2, "TIP_10"), y=0.16, text=f"Met10 ({attr156_a[9]:.2f}×)", color=BURGUNDY, border="#D8A7B1", bg="#FFF7F8"),
    ], rD2 - 0.025, side="right", fontsize=7.0)
    axD2.text(0.50, 0.972, "$\\mathrm{Met}_1$ Masked Counterfactual\n$P(\\mathrm{NP}) = 0.9187$ (Rescued TP)",
              transform=axD2.transAxes, ha="center", va="top", fontsize=7.0, fontweight="bold", color=TEAL,
              bbox=dict(boxstyle="round,pad=0.25", facecolor="#F4FBF9", edgecolor="#8BB8B2", alpha=0.96), zorder=20)

    axD1.text(0.0, 1.032, "(D) Counterfactual 3D $\\mathrm{Met}_1$ Masking Shift",
              transform=axD1.transAxes, fontsize=16.4, fontweight="bold", color="black")

    cbar_ax = fig.add_axes([0.23, 0.030, 0.54, 0.018])
    sm = plt.cm.ScalarMappable(cmap=CMAP_ATTR, norm=NORM_ATTR)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label("Residue-Level Bidirectional Co-Attention Attribution Mapped onto All-Atom 3D Structures ($\\times$ Uniform)",
                 fontsize=9.5, fontweight="bold", labelpad=5)
    cb.ax.tick_params(labelsize=8.6)
    cb.outline.set_edgecolor("#94A3B8")

    for ext in ("pdf", "png"):
        out_p = os.path.join(FIG_DIR, f"fig5b_3d_structural_motifs.{ext}")
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        print(f"Saved {out_p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
