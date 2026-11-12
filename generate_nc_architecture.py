# -*- coding: utf-8 -*-
"""
Generate Publication-Grade All-Visual Architecture Figure (fig0_architecture.pdf / .png)
for NeuroEGNN (Publication-grade vector architecture diagram).

Key design principles:
1. Zero redundant formulas — all mathematical details stay in main.tex; the figure uses
   concise, high-impact visual schematics and short module labels.
2. Large, legible typography (9.0pt - 13.0pt) designed to remain crystal clear when scaled
   to single-column print width.
3. Full-width aligned stage headers (a, b, c, d, e) with zero arrow crossings over headers.
"""

import os
import colorsys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CAND_FIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'paper', 'figures')
PAPER_FIG_DIR = _CAND_FIG_DIR if os.path.isdir(_CAND_FIG_DIR) else os.path.join(SCRIPT_DIR, 'results', 'figures')
os.makedirs(PAPER_FIG_DIR, exist_ok=True)


def mute_rgba(rgba, sat_scale=0.56, light_mix=0.06):
    """Moderately lower color saturation in HLS space while keeping clean academic hues."""
    r, g, b = mcolors.to_rgb(rgba)
    a = rgba[3] if isinstance(rgba, (tuple, list, np.ndarray)) and len(rgba) == 4 else 1.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l2 = l + light_mix * (0.95 - l)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l2, s * sat_scale)
    return (r2, g2, b2, a)


# Moderately Desaturated Scientific Palette (Publication Print Standard)
SLATE_DARK     = '#253140'
SLATE_TEXT     = '#2D3748'
SLATE_MUTED    = '#5A6677'
SLATE_LIGHT    = '#94A0B0'

NAVY_PRIMARY   = '#2C4D72'
NAVY_DEEP      = '#3D648E'
NAVY_ACCENT    = '#6083A8'
NAVY_BORDER    = '#A8BED8'
NAVY_SOFT      = '#E3ECF5'
NAVY_TINT      = '#EFF4FA'

AMBER_DARK     = '#9E6830'
AMBER_MAIN     = '#B8844A'
AMBER_TEXT     = '#7C5124'
AMBER_BORDER   = '#D4B487'
AMBER_STAGE_BD = '#DEC7A2'
AMBER_SLAB     = '#EBD6B7'
AMBER_SLAB_HI  = '#ECDBA9'
AMBER_SOFT     = '#F4EBD9'
AMBER_TINT     = '#FAF6EE'
AMBER_BG       = '#FCFBF7'

TEAL_DARK      = '#2B6B65'
TEAL_MAIN      = '#468780'
TEAL_BORDER    = '#98C4BF'
TEAL_STAGE_BD  = '#AAD0CB'
TEAL_SOFT      = '#DCECE9'
TEAL_TINT      = '#EEF6F5'
TEAL_BG        = '#F6FAF9'

INDIGO_DARK    = '#46548A'
INDIGO_BORDER  = '#A6B1D8'
INDIGO_SOFT    = '#E3E7F5'
INDIGO_TINT    = '#EEF1F9'

PURPLE_DARK    = '#5D468C'
PURPLE_MAIN    = '#7B65A8'
PURPLE_BORDER  = '#C0B4DA'
PURPLE_SOFT    = '#E8E3F4'
PURPLE_TINT    = '#F4F2FA'
PURPLE_BG      = '#F9F8FC'

CRIMSON_DARK   = '#923F52'
CRIMSON_MAIN   = '#B25B6E'
CRIMSON_TEXT   = '#7A3343'
CRIMSON_EDGE   = '#CFA0AB'
CRIMSON_BORDER = '#D9B9C0'
CRIMSON_SOFT   = '#F1E3E6'
CRIMSON_TINT   = '#F9F0F2'
CRIMSON_BG     = '#FBF8F9'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'stixsans',
    'font.size': 10.5,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def add_stage_container(ax, x, y, w, h, bg='#F8FAFC', edge='#CBD5E1', lw=1.35, radius=0.16, zorder=1):
    shadow = patches.FancyBboxPatch(
        (x + 0.03, y - 0.03), w, h,
        boxstyle=f'round,pad=0.01,rounding_size={radius}',
        facecolor='#0F172A', edgecolor='none', alpha=0.045, zorder=zorder - 0.5
    )
    ax.add_patch(shadow)
    patch = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0.01,rounding_size={radius}',
        facecolor=bg, edgecolor=edge, linewidth=lw, zorder=zorder
    )
    ax.add_patch(patch)
    return patch


def add_stage_header(ax, panel_letter, x, y, w, h, title, bg, fontsize=11.6, zorder=4):
    badge_w = 0.40
    badge = patches.FancyBboxPatch(
        (x, y), badge_w, h,
        boxstyle='round,pad=0.01,rounding_size=0.08',
        facecolor=SLATE_DARK, edgecolor='none', zorder=zorder
    )
    ax.add_patch(badge)
    ax.text(x + badge_w / 2, y + h / 2, panel_letter,
            ha='center', va='center', fontsize=13.5, fontweight='bold', color='white', zorder=zorder + 1)

    pill_x = x + badge_w + 0.08
    pill_w = w - badge_w - 0.08
    pill = patches.FancyBboxPatch(
        (pill_x, y), pill_w, h,
        boxstyle='round,pad=0.01,rounding_size=0.08',
        facecolor=bg, edgecolor='none', zorder=zorder
    )
    ax.add_patch(pill)
    ax.text(pill_x + pill_w / 2, y + h / 2, title,
            ha='center', va='center', fontsize=fontsize, fontweight='bold', color='white', zorder=zorder + 1)


def add_card(ax, x, y, w, h, title=None, title_bg=None, title_fg=SLATE_DARK,
             bg='white', edge='#CBD5E1', lw=1.15, radius=0.10, fontsize=10.8, zorder=3):
    shadow = patches.FancyBboxPatch(
        (x + 0.022, y - 0.024), w, h,
        boxstyle=f'round,pad=0.008,rounding_size={radius}',
        facecolor='#0F172A', edgecolor='none', alpha=0.055, zorder=zorder - 0.2
    )
    ax.add_patch(shadow)
    card = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0.008,rounding_size={radius}',
        facecolor=bg, edgecolor=edge, linewidth=lw, zorder=zorder
    )
    ax.add_patch(card)
    if title:
        band_h = 0.35
        if title_bg:
            band = patches.FancyBboxPatch(
                (x + 0.02, y + h - band_h - 0.01), w - 0.04, band_h,
                boxstyle='round,pad=0.005,rounding_size=0.07',
                facecolor=title_bg, edgecolor='none', zorder=zorder + 0.5
            )
            ax.add_patch(band)
        ax.text(x + w / 2, y + h - band_h / 2 - 0.01, title,
                ha='center', va='center', fontsize=fontsize, fontweight='bold',
                color=title_fg, zorder=zorder + 1)
    return card


def add_pill_box(ax, x, y, w, h, text, bg='#F8FAFC', edge='#94A3B8', fg=SLATE_TEXT,
                 lw=1.0, fontsize=9.4, fontweight='bold', radius=0.06, linespacing=1.15, zorder=5):
    p = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0.006,rounding_size={radius}',
        facecolor=bg, edgecolor=edge, linewidth=lw, zorder=zorder
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, color=fg, zorder=zorder + 1, linespacing=linespacing)
    return p


def draw_routed_arrow(ax, pts, color=SLATE_TEXT, lw=1.8, ls='-', style='-|>', mutation_scale=12, alpha=1.0, zorder=8):
    if len(pts) == 2:
        ax.annotate('', xy=pts[1], xytext=pts[0],
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                    mutation_scale=mutation_scale, shrinkA=0, shrinkB=0, alpha=alpha),
                    zorder=zorder)
    else:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        ax.plot(xs, ys, color=color, lw=lw, linestyle=ls, solid_joinstyle='round', alpha=alpha, zorder=zorder)
        ax.annotate('', xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                    mutation_scale=mutation_scale, shrinkA=0, shrinkB=0, alpha=alpha),
                    zorder=zorder)


def add_arrow_badge(ax, x, y, text, fg, bg='white', edge=None, fontsize=9.0, zorder=10):
    ec = edge if edge else fg
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=fg,
            bbox=dict(boxstyle='round,pad=0.14,rounding_size=0.08', facecolor=bg,
                      edgecolor=ec, linewidth=0.9, alpha=0.97),
            zorder=zorder)


def draw_tensor_slab(ax, x, y, w, h, n_slabs=4, dx=0.042, dy=0.042,
                     facecolor='#E7D9A0', edgecolor=AMBER_DARK, label=None, label_fg=SLATE_DARK,
                     fontsize=9.0, zorder=5):
    for k in range(n_slabs - 1, -1, -1):
        alpha = 0.52 + 0.48 * (1.0 - k / max(1, n_slabs - 1))
        slab = patches.FancyBboxPatch(
            (x + k * dx, y + k * dy), w, h,
            boxstyle='round,pad=0.004,rounding_size=0.03',
            facecolor=facecolor, edgecolor=edgecolor, linewidth=0.9, alpha=alpha,
            zorder=zorder + (n_slabs - k) * 0.1
        )
        ax.add_patch(slab)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=label_fg, zorder=zorder + 2, linespacing=1.12)


def build_architecture_figure():
    fig, ax = plt.subplots(figsize=(18.4, 9.6))
    ax.set_xlim(0, 18.4)
    ax.set_ylim(0, 9.6)
    ax.axis('off')

    # =========================================================================
    # STAGE A: Homology-Isolated Input & Biochemical Encoding (x = 0.20 .. 3.38)
    # =========================================================================
    add_stage_container(ax, 0.20, 0.28, 3.18, 9.04, bg='#F8FAFC', edge='#CBD5E1')
    add_stage_header(ax, 'A', 0.36, 8.78, 2.86, 0.38, 'Homology-Debiased Input', bg=NAVY_PRIMARY, fontsize=10.8)

    # Card A1: Prepropeptide Cleavage & Mature Sequence (y = 6.36 .. 8.60)
    add_card(ax, 0.36, 6.36, 2.86, 2.24, title='Peptide Sequence & Mask',
             title_bg='#EEF2F6', title_fg=NAVY_PRIMARY, edge='#CBD5E1', fontsize=10.0)

    # 1) Top miniature: Prepropeptide N-term Met1 signal cleavage schematic
    ax.add_patch(patches.FancyBboxPatch((0.46, 7.76), 0.58, 0.28,
                                        boxstyle='round,pad=0.005,rounding_size=0.04',
                                        facecolor='#E2E8F0', edgecolor='#94A3B8', ls='--', lw=0.9, zorder=4))
    ax.text(0.75, 7.90, r'$\mathrm{Met}_1$', ha='center', va='center',
            fontsize=8.2, fontweight='bold', color='#64748B', zorder=5)
    ax.text(0.75, 7.56, 'Cleaved', ha='center', va='center',
            fontsize=7.6, fontweight='bold', color='#64748B', zorder=5)

    # Vector scissors icon + cut line
    ax.plot([1.10, 1.10], [7.66, 8.12], color=CRIMSON_MAIN, lw=1.3, ls=':', zorder=6)
    ax.plot([0.93, 1.03], [8.12, 8.20], color=CRIMSON_MAIN, lw=1.2, zorder=6)
    ax.plot([0.93, 1.03], [8.20, 8.12], color=CRIMSON_MAIN, lw=1.2, zorder=6)
    ax.scatter([0.92, 0.92], [8.11, 8.21], s=16, facecolors='none', edgecolors=CRIMSON_MAIN, lw=1.0, zorder=6)
    ax.text(1.24, 8.16, 'KR Cut', ha='center', va='center', fontsize=7.6, fontweight='bold', color=CRIMSON_MAIN, zorder=6)

    # Mature peptide bead ribbon (S-G-G-F-A-F-S-P-R-L)
    seq_tokens = ['S', 'G', 'G', 'F', 'A', 'F', 'S', 'P', 'R', 'L']
    tok_cols = ['#64748B', '#64748B', '#64748B', CRIMSON_MAIN, TEAL_MAIN,
                CRIMSON_MAIN, '#64748B', AMBER_MAIN, NAVY_DEEP, TEAL_MAIN]
    ax.plot([1.18, 3.08], [7.90, 7.90], color='#94A3B8', lw=1.8, zorder=4)
    for idx, (aa, col) in enumerate(zip(seq_tokens, tok_cols)):
        tx = 1.18 + idx * 0.192
        bead = patches.FancyBboxPatch((tx, 7.75), 0.168, 0.30,
                                      boxstyle='round,pad=0.005,rounding_size=0.04',
                                      facecolor=col, edgecolor='white', lw=0.6, zorder=5)
        ax.add_patch(bead)
        ax.text(tx + 0.084, 7.90, aa, ha='center', va='center',
                fontsize=8.2, fontweight='bold', color='white', zorder=6)
    ax.plot([2.33, 2.33, 3.08, 3.08], [7.70, 7.64, 7.64, 7.70], color=CRIMSON_MAIN, lw=1.0, zorder=5)
    ax.text(2.70, 7.53, 'C-term Core', ha='center', va='center',
            fontsize=7.6, fontweight='bold', color=CRIMSON_MAIN, zorder=5)

    # 2) Bottom miniature: Visual Padding & Binary Mask Strip
    ax.text(1.79, 7.22, 'Padding Mask ($L_{\\max} = 50$)',
            ha='center', va='center', fontsize=8.8, fontweight='bold', color=SLATE_TEXT, zorder=5)
    mask_x0 = 0.50
    for m_i in range(12):
        is_valid = m_i < 8
        m_bg = NAVY_SOFT if is_valid else '#F1F5F9'
        m_ec = NAVY_DEEP if is_valid else '#CBD5E1'
        m_fg = NAVY_DEEP if is_valid else '#94A3B8'
        m_txt = '1' if is_valid else '0'
        bx_m = mask_x0 + m_i * 0.214
        ax.add_patch(patches.Rectangle((bx_m, 6.80), 0.190, 0.26,
                                       facecolor=m_bg, edgecolor=m_ec, lw=0.8, zorder=5))
        ax.text(bx_m + 0.095, 6.93, m_txt, ha='center', va='center',
                fontsize=8.0, fontweight='bold', color=m_fg, zorder=6)
    add_pill_box(ax, 0.50, 6.44, 2.58, 0.28,
                 'Non-$\\mathrm{Met}_1$ Debiased Subset',
                 bg=CRIMSON_TINT, edge=CRIMSON_BORDER, fg=CRIMSON_DARK, fontsize=8.4)

    # Card A2: Single-Linkage Homology Clustering (y = 3.56 .. 6.18)
    add_card(ax, 0.36, 3.56, 2.86, 2.62, title=r'Homology Split ($\tau = 0.80$)',
             title_bg='#EEF2F6', title_fg=NAVY_PRIMARY, edge='#CBD5E1', fontsize=10.0)

    clusters_vis = [
        ([(0.68, 5.40), (1.02, 5.54), (0.94, 5.16), (0.64, 5.10)], NAVY_PRIMARY, 'Train\n80%'),
        ([(1.71, 5.48), (1.99, 5.30), (1.76, 5.12)], TEAL_DARK, 'Val\n10%'),
        ([(2.54, 5.50), (2.86, 5.32), (2.66, 5.12)], CRIMSON_MAIN, 'Test\n10%'),
    ]
    for pts, c_col, c_lbl in clusters_vis:
        cx = np.mean([p[0] for p in pts])
        cy = np.mean([p[1] for p in pts])
        hull = patches.Ellipse((cx, cy), 0.66, 0.56, facecolor=c_col, alpha=0.11,
                               edgecolor=c_col, linestyle='--', lw=1.1, zorder=4)
        ax.add_patch(hull)
        for i_p in range(len(pts)):
            for j_p in range(i_p + 1, len(pts)):
                ax.plot([pts[i_p][0], pts[j_p][0]], [pts[i_p][1], pts[j_p][1]],
                        color=c_col, lw=1.2, alpha=0.65, zorder=4)
        for px, py in pts:
            ax.scatter([px], [py], s=34, color=c_col, edgecolor='white', linewidth=0.8, zorder=5)
        ax.text(cx, 4.74, c_lbl, ha='center', va='center', fontsize=8.4, fontweight='bold',
                color=c_col, linespacing=1.05, zorder=5)

    for sx in [1.33, 2.23]:
        ax.plot([sx, sx], [4.58, 5.68], color=CRIMSON_MAIN, lw=1.1, ls='--', alpha=0.7, zorder=4)

    add_pill_box(ax, 0.50, 4.06, 2.58, 0.34,
                 '7,050 Homology Clusters',
                 bg=NAVY_TINT, edge=NAVY_BORDER, fg=NAVY_PRIMARY, fontsize=8.8)
    add_pill_box(ax, 0.50, 3.66, 2.58, 0.32,
                 'Zero Cross-Split Leakage',
                 bg='#F8FAFC', edge='#CBD5E1', fg=SLATE_MUTED, fontsize=8.5)

    # Card A3: 26-D Residue Biochemical Feature Encoding (y = 0.46 .. 3.36)
    add_card(ax, 0.36, 0.46, 2.86, 2.90, title='26-D Residue Encoding',
             title_bg='#EEF2F6', title_fg=NAVY_PRIMARY, edge='#CBD5E1', fontsize=10.0)

    fx0, fy0, fw_oh, fw_pc, fh_m = 0.52, 2.22, 1.48, 0.92, 0.60
    ax.add_patch(patches.Rectangle((fx0, fy0), fw_oh, fh_m, facecolor=NAVY_TINT, edgecolor=NAVY_DEEP, lw=0.9, zorder=4))
    n_rows, n_oh = 6, 12
    cw_oh, ch_r = fw_oh / n_oh, fh_m / n_rows
    hot_cols = [1, 5, 3, 9, 0, 7]
    for r in range(n_rows):
        c = hot_cols[r]
        ax.add_patch(patches.Rectangle((fx0 + c * cw_oh + 0.01, fy0 + r * ch_r + 0.01),
                                       cw_oh - 0.02, ch_r - 0.02, facecolor=NAVY_DEEP, edgecolor='none', zorder=5))
    pc_vals = np.array([
        [0.85, 0.30, 0.65, 0.20, 0.75],
        [0.25, 0.80, 0.40, 0.90, 0.35],
        [0.70, 0.45, 0.85, 0.15, 0.60],
        [0.40, 0.75, 0.30, 0.80, 0.50],
        [0.90, 0.20, 0.75, 0.45, 0.85],
        [0.55, 0.65, 0.50, 0.70, 0.40],
    ])
    cw_pc = fw_pc / 5.0
    for r in range(n_rows):
        for c in range(5):
            ax.add_patch(patches.Rectangle((fx0 + fw_oh + 0.08 + c * cw_pc, fy0 + r * ch_r),
                                           cw_pc, ch_r, facecolor=mute_rgba(plt.cm.YlGnBu(pc_vals[r, c] * 0.8 + 0.12)),
                                           edgecolor='white', lw=0.3, zorder=4))
    ax.add_patch(patches.Rectangle((fx0 + fw_oh + 0.08, fy0), fw_pc, fh_m,
                                   facecolor='none', edgecolor=TEAL_DARK, lw=0.9, zorder=5))
    ax.text(fx0 + fw_oh / 2, fy0 - 0.15, '21-D One-Hot', ha='center', va='center',
            fontsize=8.2, fontweight='bold', color=NAVY_DEEP, zorder=5)
    ax.text(fx0 + fw_oh + 0.08 + fw_pc / 2, fy0 - 0.15, '5-D PhysChem', ha='center', va='center',
            fontsize=8.2, fontweight='bold', color=TEAL_DARK, zorder=5)

    p_tags = ['Hydropathy', 'Polarity', 'Volume', 'Charge', 'Flexibility']
    for t_i, p_lbl in enumerate(p_tags):
        row_t = t_i // 3
        col_t = t_i % 3
        tx_p = 0.48 + col_t * 0.89 if row_t == 0 else 0.88 + col_t * 0.94
        ty_p = 1.66 if row_t == 0 else 1.34
        add_pill_box(ax, tx_p, ty_p, 0.84, 0.25, p_lbl,
                     bg=TEAL_TINT, edge=TEAL_BORDER, fg=TEAL_DARK, fontsize=7.6, lw=0.8)

    add_pill_box(ax, 0.52, 0.68, 1.24, 0.48,
                 'Linear $26\\to 128$\nLN + SiLU',
                 bg='#F8FAFC', edge=TEAL_MAIN, fg=SLATE_TEXT, fontsize=8.0)
    draw_routed_arrow(ax, [(1.78, 0.92), (2.06, 0.92)], color=TEAL_DARK, lw=1.5, mutation_scale=9)
    draw_tensor_slab(ax, 2.08, 0.62, 0.84, 0.52, n_slabs=3, dx=0.035, dy=0.035,
                     facecolor=TEAL_SOFT, edgecolor=TEAL_DARK,
                     label='$\\mathbf{H}_{\\mathrm{node}}$\n$128$-D', label_fg=TEAL_DARK, fontsize=8.4)

    # =========================================================================
    # STAGE B: 1D Evolutionary Semantic Pathway (Top Middle: x = 3.90 .. 10.30, y = 5.56 .. 9.32)
    # =========================================================================
    add_stage_container(ax, 3.90, 5.56, 6.40, 3.76, bg=AMBER_BG, edge=AMBER_STAGE_BD)
    add_stage_header(ax, 'B', 4.06, 8.78, 6.08, 0.38,
                     '1D Evolutionary Pathway (Frozen ESM-2 Language Model)', bg=AMBER_DARK, fontsize=10.8)

    # Card B1: Frozen ESM-2 (x = 4.06 .. 7.14, y = 5.74 .. 8.60)
    add_card(ax, 4.06, 5.74, 3.08, 2.86, title='Frozen ESM-2 (650M)',
             title_bg=AMBER_SOFT, title_fg=AMBER_DARK, edge=AMBER_BORDER, fontsize=10.0)

    res_xs = np.linspace(4.36, 6.84, 8)
    for rx in res_xs:
        ax.scatter([rx], [7.80], s=36, color=AMBER_MAIN, edgecolor='white', lw=0.7, zorder=6)
    ax.plot([4.36, 6.84], [7.80, 7.80], color=AMBER_BORDER, lw=1.6, zorder=5)
    for i_a, j_a, h_arc, lw_a in [(0, 3, 0.26, 1.1), (1, 6, 0.38, 1.5), (2, 7, 0.42, 1.7), (4, 7, 0.28, 1.2)]:
        x_mid = 0.5 * (res_xs[i_a] + res_xs[j_a])
        w_arc = res_xs[j_a] - res_xs[i_a]
        arc = patches.Arc((x_mid, 7.82), w_arc, h_arc, angle=0, theta1=0, theta2=180,
                          color=AMBER_DARK, lw=lw_a, alpha=0.82, zorder=5)
        ax.add_patch(arc)

    for ly, ltxt in [(7.24, 'Transformer Block 1 (20 Heads)'),
                     (6.82, 'Transformer Block 2 (20 Heads)'),
                     (6.26, 'Transformer Block 33 (20 Heads)')]:
        add_pill_box(ax, 4.22, ly, 2.76, 0.34, ltxt,
                     bg=AMBER_TINT, edge=AMBER_BORDER, fg=AMBER_TEXT, fontsize=8.4)
    ax.text(5.60, 6.68, r'$\bullet\;\bullet\;\bullet$', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=AMBER_DARK, zorder=6)

    add_pill_box(ax, 4.22, 5.82, 1.32, 0.34, r'Contact Map $\mathbf{C}$',
                 bg=AMBER_SOFT, edge=AMBER_DARK, fg=AMBER_TEXT, fontsize=8.2)
    add_pill_box(ax, 5.64, 5.82, 1.34, 0.34, r'Embeddings $\mathbf{E}$',
                 bg=AMBER_SOFT, edge=AMBER_DARK, fg=AMBER_TEXT, fontsize=8.2)

    # Card B2: 1D Bottleneck Projection (x = 7.50 .. 10.14, y = 5.74 .. 8.60)
    add_card(ax, 7.50, 5.74, 2.64, 2.86, title='1D Semantic Projection',
             title_bg=AMBER_SOFT, title_fg=AMBER_DARK, edge=AMBER_BORDER, fontsize=10.0)

    draw_tensor_slab(ax, 7.62, 6.82, 0.58, 1.18, n_slabs=3, dx=0.035, dy=0.035,
                     facecolor=AMBER_SLAB, edgecolor=AMBER_DARK,
                     label='$\\mathbf{E}$\n$1280$-D', label_fg=AMBER_TEXT, fontsize=8.2)

    funnel_pts = [(8.32, 7.98), (9.00, 7.68), (9.00, 7.14), (8.32, 6.84)]
    ax.add_patch(patches.Polygon(funnel_pts, closed=True, facecolor=AMBER_SOFT,
                                 edgecolor=AMBER_MAIN, lw=1.0, zorder=4))
    ax.text(8.66, 7.41, 'Linear\nLN + SiLU\nDropout', ha='center', va='center',
            fontsize=7.6, fontweight='bold', color=AMBER_TEXT, linespacing=1.15, zorder=5)

    draw_tensor_slab(ax, 9.10, 7.02, 0.80, 0.78, n_slabs=4, dx=0.04, dy=0.04,
                     facecolor=AMBER_SLAB_HI, edgecolor=AMBER_DARK,
                     label='$\\mathbf{H}_{\\mathrm{seq}}$\n$128$-D', label_fg=AMBER_TEXT, fontsize=8.8)

    ax.text(8.82, 6.42, 'Evolutionary Sequence Motifs',
            ha='center', va='center', fontsize=8.4, fontweight='bold', color=SLATE_TEXT, zorder=5)
    for mx_t, m_lbl in [(7.62, '-FXPRL'), (8.48, '-FMRF'), (9.34, 'KR / RR')]:
        add_pill_box(ax, mx_t, 5.88, 0.78, 0.36, m_lbl,
                     bg=AMBER_TINT, edge=AMBER_MAIN, fg=AMBER_TEXT, fontsize=8.0)

    draw_routed_arrow(ax, [(7.14, 7.42), (7.50, 7.42)], color=AMBER_DARK, lw=2.0)
    ax.text(7.32, 7.68, r'$\mathbf{E}$', ha='center', va='center', fontsize=9.8, fontweight='bold', color=AMBER_DARK)

    # =========================================================================
    # STAGE C: 3D Conformational Graph & E(3)-EGNN (Bottom Middle: x = 3.90 .. 10.30, y = 0.28 .. 4.96)
    # =========================================================================
    add_stage_container(ax, 3.90, 0.28, 6.40, 4.68, bg=TEAL_BG, edge=TEAL_STAGE_BD)
    add_stage_header(ax, 'C', 4.06, 4.44, 6.08, 0.38,
                     '3D Conformational Graph & E(3)-Equivariant GNN', bg=TEAL_DARK, fontsize=10.8)

    # Card C1: 3D Structure & Hybrid Graph Construction (x = 4.06 .. 6.94, y = 0.46 .. 4.28)
    add_card(ax, 4.06, 0.46, 2.88, 3.82, title='3D Structure & Hybrid Graph',
             title_bg=TEAL_SOFT, title_fg=TEAL_DARK, edge=TEAL_MAIN, fontsize=10.0)

    idx_g = np.arange(10)
    d_g = np.abs(idx_g[:, None] - idx_g[None, :])
    cmap_c = np.exp(-d_g / 2.1)
    cmap_c[2, 8] = cmap_c[8, 2] = 0.88
    cmap_c[1, 7] = cmap_c[7, 1] = 0.80
    mx0, my0, msz = 4.18, 2.84, 0.90
    cs = msz / 10.0
    for r_i in range(10):
        for c_i in range(10):
            ax.add_patch(patches.Rectangle(
                (mx0 + c_i * cs, my0 + (9 - r_i) * cs), cs, cs,
                facecolor=mute_rgba(plt.cm.YlGnBu(cmap_c[r_i, c_i] * 0.85 + 0.10)), edgecolor='none', zorder=4
            ))
    ax.add_patch(patches.Rectangle((mx0, my0), msz, msz, facecolor='none', edgecolor=NAVY_PRIMARY, lw=0.9, zorder=5))
    ax.text(mx0 + msz / 2, my0 - 0.16, r'Contact Map $\mathbf{C}$', ha='center', va='center',
            fontsize=8.2, fontweight='bold', color=NAVY_PRIMARY, zorder=5)

    draw_routed_arrow(ax, [(5.14, 3.28), (5.54, 3.28)], color=TEAL_DARK, lw=1.6, mutation_scale=9)
    ax.text(5.34, 3.55, 'MDS /\nESMFold', ha='center', va='center',
            fontsize=7.6, fontweight='bold', color=TEAL_DARK, linespacing=1.06, zorder=5)

    pts_3d = [
        (5.66, 3.68), (5.94, 3.78), (6.26, 3.70), (6.54, 3.50),
        (6.64, 3.20), (6.40, 2.96), (6.04, 2.92), (5.74, 3.12), (5.90, 3.40)
    ]
    for u_i, v_i in [(0, 7), (1, 8), (2, 6), (3, 5)]:
        ax.plot([pts_3d[u_i][0], pts_3d[v_i][0]], [pts_3d[u_i][1], pts_3d[v_i][1]],
                color=CRIMSON_MAIN, linestyle='--', linewidth=1.1, alpha=0.85, zorder=4)
    ax.plot([p[0] for p in pts_3d], [p[1] for p in pts_3d], color=TEAL_DARK, linewidth=2.2, zorder=5)
    for idx_p, (px, py) in enumerate(pts_3d):
        ncol = CRIMSON_MAIN if idx_p in (2, 6, 8) else TEAL_MAIN
        ax.scatter([px], [py], s=36, color=ncol, edgecolor='white', linewidth=0.8, zorder=6)
    ax.text(6.16, 2.68, r'3D $\mathrm{C}_\alpha$ Coords $\mathbf{X}^{(0)}$', ha='center', va='center',
            fontsize=8.2, fontweight='bold', color=TEAL_DARK, zorder=5)

    ax.text(5.50, 2.32, 'Hybrid Graph Edges (3-Way Union)',
            ha='center', va='center', fontsize=8.4, fontweight='bold', color=SLATE_TEXT, zorder=5)

    # Mini-card 1: Backbone
    ax.add_patch(patches.FancyBboxPatch((4.18, 1.22), 0.80, 0.92,
                                        boxstyle='round,pad=0.006,rounding_size=0.05',
                                        facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=0.9, zorder=4))
    bb_pts = [(4.30, 1.74), (4.48, 1.94), (4.68, 1.74), (4.86, 1.94)]
    ax.plot([p[0] for p in bb_pts], [p[1] for p in bb_pts], color=TEAL_DARK, lw=1.9, zorder=5)
    ax.plot([bb_pts[0][0], bb_pts[2][0]], [bb_pts[0][1], bb_pts[2][1]], color=TEAL_MAIN, lw=1.0, ls='--', zorder=4.5)
    for px, py in bb_pts:
        ax.scatter([px], [py], s=24, color=TEAL_MAIN, edgecolor='white', lw=0.6, zorder=6)
    ax.text(4.58, 1.42, 'Backbone\nBonds', ha='center', va='center',
            fontsize=7.6, fontweight='bold', color=TEAL_DARK, linespacing=1.08, zorder=5)

    # Mini-card 2: Spatial Cutoff
    ax.add_patch(patches.FancyBboxPatch((5.10, 1.22), 0.80, 0.92,
                                        boxstyle='round,pad=0.006,rounding_size=0.05',
                                        facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=0.9, zorder=4))
    ax.add_patch(patches.Circle((5.50, 1.82), 0.23, facecolor=CRIMSON_SOFT, edgecolor=CRIMSON_MAIN,
                                ls='--', lw=0.9, alpha=0.7, zorder=4.5))
    ax.scatter([5.50], [1.82], s=28, color=CRIMSON_MAIN, edgecolor='white', lw=0.6, zorder=6)
    for nx_s, ny_s in [(5.34, 1.94), (5.66, 1.90), (5.56, 1.65)]:
        ax.plot([5.50, nx_s], [1.82, ny_s], color=CRIMSON_MAIN, lw=1.0, zorder=5)
        ax.scatter([nx_s], [ny_s], s=18, color=NAVY_PRIMARY, edgecolor='white', lw=0.5, zorder=6)
    ax.text(5.50, 1.42, '$<10\\,\\mathrm{\\AA}$\nSphere', ha='center', va='center',
            fontsize=7.6, fontweight='bold', color=CRIMSON_DARK, linespacing=1.08, zorder=5)

    # Mini-card 3: k-NN Spatial
    ax.add_patch(patches.FancyBboxPatch((6.02, 1.22), 0.80, 0.92,
                                        boxstyle='round,pad=0.006,rounding_size=0.05',
                                        facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=0.9, zorder=4))
    ax.scatter([6.42], [1.82], s=28, color=NAVY_DEEP, edgecolor='white', lw=0.6, zorder=6)
    for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        kx = 6.42 + 0.21 * np.cos(ang)
        ky = 1.82 + 0.17 * np.sin(ang)
        ax.plot([6.42, kx], [1.82, ky], color=NAVY_ACCENT, lw=0.9, ls=':', zorder=5)
        ax.scatter([kx], [ky], s=16, color=NAVY_ACCENT, edgecolor='white', lw=0.4, zorder=6)
    ax.text(6.42, 1.42, 'Spatial\n$8$-NN', ha='center', va='center',
            fontsize=7.6, fontweight='bold', color=NAVY_PRIMARY, linespacing=1.08, zorder=5)

    add_pill_box(ax, 4.18, 0.60, 2.64, 0.46,
                 r'Node Fusion: $\mathbf{H}_{\mathrm{node}} \oplus \mathbf{H}_{\mathrm{seq}}$',
                 bg=TEAL_TINT, edge=TEAL_MAIN, fg=TEAL_DARK, fontsize=8.6)

    # Card C2: 4-Layer E(3)-Equivariant GNN & Conformational Relaxation (x = 7.20 .. 10.14, y = 0.46 .. 4.28)
    add_card(ax, 7.20, 0.46, 2.94, 3.82, title=r'4-Layer $E(3)$-Equivariant GNN',
             title_bg=TEAL_SOFT, title_fg=TEAL_DARK, edge=TEAL_MAIN, fontsize=10.0)

    # Top Visual Box in C2: Equivariant Message Passing & Radial Force Update
    ax.add_patch(patches.FancyBboxPatch((7.34, 2.46), 2.66, 1.36,
                                        boxstyle='round,pad=0.008,rounding_size=0.06',
                                        facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=0.9, zorder=3.5))
    ax.text(8.67, 3.68, 'Equivariant Coordinate Forces',
            ha='center', va='center', fontsize=8.6, fontweight='bold', color=SLATE_TEXT, zorder=5)

    ci_x, ci_y = 7.98, 2.98
    ax.add_patch(patches.Circle((ci_x, ci_y), 0.43, facecolor=TEAL_TINT, edgecolor=TEAL_MAIN,
                                ls=':', lw=0.9, alpha=0.65, zorder=4))
    nbr_nodes = [(7.50, 3.26, '$j_1$'), (7.52, 2.68, '$j_2$'), (8.46, 3.28, '$j_3$')]
    for nx, ny, nlbl in nbr_nodes:
        ax.plot([ci_x, nx], [ci_y, ny], color='#94A3B8', lw=1.1, zorder=4.5)
        fx_v = ci_x + 0.52 * (nx - ci_x)
        fy_v = ci_y + 0.52 * (ny - ci_y)
        ax.annotate('', xy=(fx_v, fy_v), xytext=(ci_x, ci_y),
                    arrowprops=dict(arrowstyle='->', color=CRIMSON_MAIN, lw=1.5, mutation_scale=8), zorder=6)
        ax.scatter([nx], [ny], s=48, color=NAVY_PRIMARY, edgecolor='white', lw=0.7, zorder=6)
        ax.text(nx, ny, nlbl, ha='center', va='center', fontsize=7.4, fontweight='bold', color='white', zorder=7)

    ax.scatter([ci_x], [ci_y], s=74, color=TEAL_DARK, edgecolor='white', lw=0.9, zorder=7)
    ax.text(ci_x, ci_y, '$i$', ha='center', va='center', fontsize=8.4, fontweight='bold', color='white', zorder=8)
    cnew_x, cnew_y = 8.44, 2.68
    ax.annotate('', xy=(cnew_x, cnew_y), xytext=(ci_x, ci_y),
                arrowprops=dict(arrowstyle='->', color=AMBER_DARK, lw=1.7, ls='--', mutation_scale=10), zorder=7)
    ax.scatter([cnew_x], [cnew_y], s=56, color=AMBER_MAIN, edgecolor='white', lw=0.8, zorder=7)

    # Visual O(3) 3D Coordinate Axes Badge (sits cleanly below the top title!)
    ax.add_patch(patches.FancyBboxPatch((8.94, 2.54), 0.96, 0.98,
                                        boxstyle='round,pad=0.006,rounding_size=0.05',
                                        facecolor=TEAL_TINT, edgecolor=TEAL_MAIN, lw=0.9, zorder=4))
    ox, oy = 9.32, 3.16
    ax.annotate('', xy=(ox + 0.28, oy + 0.07), xytext=(ox, oy),
                arrowprops=dict(arrowstyle='->', color=CRIMSON_MAIN, lw=1.3, mutation_scale=7), zorder=6)
    ax.annotate('', xy=(ox - 0.12, oy + 0.24), xytext=(ox, oy),
                arrowprops=dict(arrowstyle='->', color=TEAL_DARK, lw=1.3, mutation_scale=7), zorder=6)
    ax.annotate('', xy=(ox - 0.16, oy - 0.15), xytext=(ox, oy),
                arrowprops=dict(arrowstyle='->', color=NAVY_DEEP, lw=1.3, mutation_scale=7), zorder=6)
    ax.text(9.42, 2.76, '$E(3)$ / $O(3)$\nEquivariant',
            ha='center', va='center', fontsize=7.8, fontweight='bold', color=TEAL_DARK, linespacing=1.10, zorder=6)

    # Bottom Visual Box in C2: 4-Layer Conformational Compaction Trajectory
    ax.add_patch(patches.FancyBboxPatch((7.34, 1.18), 2.66, 1.16,
                                        boxstyle='round,pad=0.008,rounding_size=0.06',
                                        facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=0.9, zorder=3.5))
    ax.text(8.67, 2.18, r'3D Conformational Compaction ($\mathbf{X}^{(0)} \to \mathbf{X}^{(4)}$)',
            ha='center', va='center', fontsize=8.3, fontweight='bold', color=TEAL_DARK, zorder=5)

    t_curve = np.linspace(0, 1, 7)
    x0_pts = np.column_stack([7.50 + 1.46 * t_curve, 1.62 + 0.36 * np.sin(np.pi * t_curve)])
    x4_pts = np.column_stack([7.66 + 1.14 * t_curve, 1.54 + 0.46 * np.sin(np.pi * t_curve)])
    ax.plot(x0_pts[:, 0], x0_pts[:, 1], color='#94A3B8', lw=1.4, ls='--', zorder=4)
    ax.scatter(x0_pts[:, 0], x0_pts[:, 1], s=20, color='#94A3B8', edgecolor='white', lw=0.4, zorder=4.5)
    ax.plot(x4_pts[:, 0], x4_pts[:, 1], color=TEAL_DARK, lw=2.2, zorder=5)
    ax.scatter(x4_pts[:, 0], x4_pts[:, 1], s=30, color=TEAL_MAIN, edgecolor='white', lw=0.6, zorder=6)
    for k_idx in [0, 6]:
        ax.annotate('', xy=(x4_pts[k_idx, 0], x4_pts[k_idx, 1]),
                    xytext=(x0_pts[k_idx, 0], x0_pts[k_idx, 1]),
                    arrowprops=dict(arrowstyle='->', color=AMBER_DARK, lw=1.3, mutation_scale=7), zorder=7)
    ax.annotate('', xy=(x4_pts[-1, 0], 1.41), xytext=(x4_pts[0, 0], 1.41),
                arrowprops=dict(arrowstyle='<->', color=CRIMSON_MAIN, lw=1.1, mutation_scale=7), zorder=6)
    ax.text(8.23, 1.27, 'N–C Compaction',
            ha='center', va='center', fontsize=7.6, fontweight='bold', color=CRIMSON_MAIN, zorder=6)

    draw_tensor_slab(ax, 9.06, 1.32, 0.78, 0.58, n_slabs=3, dx=0.032, dy=0.032,
                     facecolor=TEAL_SOFT, edgecolor=TEAL_DARK,
                     label='$\\mathbf{H}_{\\mathrm{graph}}$\n$128$-D', label_fg=TEAL_DARK, fontsize=8.0)

    add_pill_box(ax, 7.34, 0.60, 2.66, 0.46,
                 '4 Equivariant Message-Passing Layers',
                 bg=TEAL_TINT, edge=TEAL_MAIN, fg=TEAL_DARK, fontsize=8.4)

    draw_routed_arrow(ax, [(6.94, 2.35), (7.20, 2.35)], color=TEAL_DARK, lw=2.0)

    # =========================================================================
    # STAGE D: Symmetric Bidirectional Co-Attention (x = 10.82 .. 14.48, y = 0.28 .. 9.32)
    # =========================================================================
    add_stage_container(ax, 10.82, 0.28, 3.66, 9.04, bg=PURPLE_BG, edge=PURPLE_BORDER)
    add_stage_header(ax, 'D', 10.98, 8.78, 3.34, 0.38,
                     'Gated Bidirectional Co-Attention', bg=PURPLE_DARK, fontsize=10.8)

    # Card D1 (TOP): Visual Seq -> Graph Cross-Attention (y = 6.42 .. 8.60)
    add_card(ax, 10.98, 6.42, 3.34, 2.18,
             title=r'Seq $\to$ Graph Cross-Attention (4 Heads)',
             title_bg=PURPLE_SOFT, title_fg=PURPLE_DARK, edge=PURPLE_BORDER, fontsize=9.6)

    add_pill_box(ax, 11.10, 7.80, 0.92, 0.30, 'Query: Seq',
                 bg=AMBER_SOFT, edge=AMBER_MAIN, fg=AMBER_TEXT, fontsize=7.8)
    add_pill_box(ax, 11.10, 7.42, 0.92, 0.30, 'Key: Graph',
                 bg=TEAL_SOFT, edge=TEAL_MAIN, fg=TEAL_DARK, fontsize=7.8)
    add_pill_box(ax, 11.10, 6.96, 0.92, 0.30, 'Value: Graph',
                 bg=TEAL_SOFT, edge=TEAL_MAIN, fg=TEAL_DARK, fontsize=7.8)

    draw_routed_arrow(ax, [(12.02, 7.95), (12.22, 7.76)], color=AMBER_DARK, lw=1.3, mutation_scale=7)
    draw_routed_arrow(ax, [(12.02, 7.57), (12.22, 7.57)], color=TEAL_DARK, lw=1.3, mutation_scale=7)

    s2g_x, s2g_y, s2g_sz = 12.24, 7.20, 0.80
    s2g_mat = np.eye(8) * 0.35 + 0.10
    s2g_mat[:, 2] = 0.85
    s2g_mat[:, 6] = 0.92
    s2g_mat[6, 2] = s2g_mat[2, 6] = 0.96
    cs_a = s2g_sz / 8.0
    for r_i in range(8):
        for c_i in range(8):
            ax.add_patch(patches.Rectangle(
                (s2g_x + c_i * cs_a, s2g_y + (7 - r_i) * cs_a), cs_a, cs_a,
                facecolor=mute_rgba(plt.cm.Purples(min(1.0, s2g_mat[r_i, c_i]))), edgecolor='none', zorder=4
            ))
    ax.add_patch(patches.Rectangle((s2g_x, s2g_y), s2g_sz, s2g_sz, facecolor='none', edgecolor=PURPLE_DARK, lw=0.9, zorder=5))
    ax.text(s2g_x + s2g_sz / 2, s2g_y + s2g_sz + 0.11, r'Attn $\mathbf{A}_{\mathrm{s}\to\mathrm{g}}$',
            ha='center', va='center', fontsize=8.0, fontweight='bold', color=PURPLE_DARK, zorder=5)

    ax.scatter([13.28], [7.60], s=95, color=PURPLE_DARK, edgecolor='white', lw=0.8, zorder=6)
    ax.text(13.28, 7.60, r'$\otimes$', ha='center', va='center', fontsize=9.0, fontweight='bold', color='white', zorder=7)
    draw_routed_arrow(ax, [(13.04, 7.60), (13.15, 7.60)], color=PURPLE_DARK, lw=1.3, mutation_scale=7)
    draw_routed_arrow(ax, [(12.02, 7.04), (13.28, 7.04), (13.28, 7.46)], color=TEAL_DARK, lw=1.3, mutation_scale=7)
    draw_routed_arrow(ax, [(13.41, 7.60), (13.54, 7.60)], color=PURPLE_DARK, lw=1.3, mutation_scale=7)

    draw_tensor_slab(ax, 13.56, 7.28, 0.58, 0.58, n_slabs=3, dx=0.03, dy=0.03,
                     facecolor=PURPLE_SOFT, edgecolor=PURPLE_DARK,
                     label='$\\mathbf{Z}_{\\mathrm{s}\\to\\mathrm{g}}$', label_fg=PURPLE_DARK, fontsize=8.6)

    add_pill_box(ax, 11.12, 6.52, 3.06, 0.32,
                 'Captures Long-Range 3D Spatial Contacts',
                 bg=PURPLE_TINT, edge=PURPLE_BORDER, fg=PURPLE_DARK, fontsize=8.3)

    # Card D2 (CENTER): Visual Dual-Modality Gating & Pre-LN FFN Circuit (y = 3.04 .. 6.02)
    add_card(ax, 10.98, 3.04, 3.34, 2.98,
             title='Adaptive Dual-Modality Gating & FFN',
             title_bg=INDIGO_SOFT, title_fg=INDIGO_DARK, edge=INDIGO_BORDER, fontsize=9.6)

    # Visual Sigmoid Gate Controller block at top center
    ax.add_patch(patches.FancyBboxPatch((11.48, 5.18), 2.34, 0.42,
                                        boxstyle='round,pad=0.006,rounding_size=0.06',
                                        facecolor=INDIGO_TINT, edgecolor=INDIGO_DARK, lw=1.0, zorder=5))
    sig_t = np.linspace(-3, 3, 25)
    sig_x = 11.76 + 0.15 * (sig_t / 3.0)
    sig_y = 5.39 + 0.13 * (1.0 / (1.0 + np.exp(-sig_t)) - 0.5) * 2.0
    ax.plot(sig_x, sig_y, color=CRIMSON_MAIN, lw=1.7, zorder=6)
    ax.plot([11.60, 11.92], [5.39, 5.39], color='#94A3B8', lw=0.7, zorder=5.5)
    ax.text(12.88, 5.39, '2-Channel Sigmoid Gate',
            ha='center', va='center', fontsize=8.5, fontweight='bold', color=INDIGO_DARK, zorder=6)

    # Left & Right Gated Multiplier Branches
    add_pill_box(ax, 11.10, 4.52, 1.18, 0.42,
                 'Gated Seq\nBranch',
                 bg=PURPLE_SOFT, edge=PURPLE_MAIN, fg=PURPLE_DARK, fontsize=8.0)
    add_pill_box(ax, 13.02, 4.52, 1.18, 0.42,
                 'Gated Graph\nBranch',
                 bg=NAVY_SOFT, edge=NAVY_DEEP, fg=NAVY_PRIMARY, fontsize=8.0)

    draw_routed_arrow(ax, [(11.85, 5.18), (11.69, 4.94)], color=PURPLE_DARK, lw=1.3, mutation_scale=8)
    ax.text(11.55, 5.08, r'$\alpha_{\mathrm{seq}}$', ha='right', va='center', fontsize=8.4, fontweight='bold', color=PURPLE_DARK)
    draw_routed_arrow(ax, [(13.45, 5.18), (13.61, 4.94)], color=NAVY_PRIMARY, lw=1.3, mutation_scale=8)
    ax.text(13.75, 5.08, r'$\alpha_{\mathrm{graph}}$', ha='left', va='center', fontsize=8.4, fontweight='bold', color=NAVY_PRIMARY)

    # Central Summation node (+)
    ax.scatter([12.65], [4.73], s=145, color=INDIGO_DARK, edgecolor='white', lw=1.0, zorder=6)
    ax.text(12.65, 4.73, r'$\oplus$', ha='center', va='center', fontsize=10.5, fontweight='bold', color='white', zorder=7)
    draw_routed_arrow(ax, [(12.28, 4.73), (12.49, 4.73)], color=PURPLE_DARK, lw=1.4, mutation_scale=8)
    draw_routed_arrow(ax, [(13.02, 4.73), (12.81, 4.73)], color=NAVY_PRIMARY, lw=1.4, mutation_scale=8)

    # Visual Pre-LN + FFN + Fused Tensor
    draw_routed_arrow(ax, [(12.65, 4.53), (12.65, 4.28)], color=INDIGO_DARK, lw=1.5, mutation_scale=8)
    add_pill_box(ax, 11.14, 3.84, 1.86, 0.42,
                 'LayerNorm + 2-Layer FFN\n(Residual Skip)',
                 bg=INDIGO_TINT, edge=INDIGO_DARK, fg=INDIGO_DARK, fontsize=7.9)
    draw_routed_arrow(ax, [(13.00, 4.05), (13.30, 4.05)], color=INDIGO_DARK, lw=1.5, mutation_scale=8)
    draw_tensor_slab(ax, 13.34, 3.76, 0.74, 0.56, n_slabs=3, dx=0.032, dy=0.032,
                     facecolor=INDIGO_SOFT, edgecolor=INDIGO_DARK,
                     label='$\\mathbf{H}_{\\mathrm{fuse}}$\n$128$-D', label_fg=INDIGO_DARK, fontsize=8.2)

    add_pill_box(ax, 11.12, 3.16, 3.06, 0.34,
                 'Dynamic Sequence–Structure Alignment',
                 bg='#F8FAFC', edge='#CBD5E1', fg=SLATE_TEXT, fontsize=8.3)

    # Card D3 (BOTTOM): Visual Graph -> Seq Cross-Attention (y = 0.46 .. 2.66)
    add_card(ax, 10.98, 0.46, 3.34, 2.20,
             title=r'Graph $\to$ Seq Cross-Attention (4 Heads)',
             title_bg=NAVY_SOFT, title_fg=NAVY_PRIMARY, edge=NAVY_BORDER, fontsize=9.6)

    add_pill_box(ax, 11.10, 1.84, 0.92, 0.30, 'Query: Graph',
                 bg=TEAL_SOFT, edge=TEAL_MAIN, fg=TEAL_DARK, fontsize=7.8)
    add_pill_box(ax, 11.10, 1.46, 0.92, 0.30, 'Key: Seq',
                 bg=AMBER_SOFT, edge=AMBER_MAIN, fg=AMBER_TEXT, fontsize=7.8)
    add_pill_box(ax, 11.10, 1.00, 0.92, 0.30, 'Value: Seq',
                 bg=AMBER_SOFT, edge=AMBER_MAIN, fg=AMBER_TEXT, fontsize=7.8)

    draw_routed_arrow(ax, [(12.02, 1.99), (12.22, 1.80)], color=TEAL_DARK, lw=1.3, mutation_scale=7)
    draw_routed_arrow(ax, [(12.02, 1.61), (12.22, 1.61)], color=AMBER_DARK, lw=1.3, mutation_scale=7)

    g2s_x, g2s_y = 12.24, 1.24
    g2s_mat = np.eye(8) * 0.65 + np.roll(np.eye(8), 1, axis=1) * 0.30 + 0.08
    g2s_mat[:, 7] = 0.78
    for r_i in range(8):
        for c_i in range(8):
            ax.add_patch(patches.Rectangle(
                (g2s_x + c_i * cs_a, g2s_y + (7 - r_i) * cs_a), cs_a, cs_a,
                facecolor=mute_rgba(plt.cm.Blues(min(1.0, g2s_mat[r_i, c_i]))), edgecolor='none', zorder=4
            ))
    ax.add_patch(patches.Rectangle((g2s_x, g2s_y), s2g_sz, s2g_sz, facecolor='none', edgecolor=NAVY_PRIMARY, lw=0.9, zorder=5))
    ax.text(g2s_x + s2g_sz / 2, g2s_y + s2g_sz + 0.11, r'Attn $\mathbf{A}_{\mathrm{g}\to\mathrm{s}}$',
            ha='center', va='center', fontsize=8.0, fontweight='bold', color=NAVY_PRIMARY, zorder=5)

    ax.scatter([13.28], [1.64], s=95, color=NAVY_PRIMARY, edgecolor='white', lw=0.8, zorder=6)
    ax.text(13.28, 1.64, r'$\otimes$', ha='center', va='center', fontsize=9.0, fontweight='bold', color='white', zorder=7)
    draw_routed_arrow(ax, [(13.04, 1.64), (13.15, 1.64)], color=NAVY_PRIMARY, lw=1.3, mutation_scale=7)
    draw_routed_arrow(ax, [(12.02, 1.08), (13.28, 1.08), (13.28, 1.50)], color=AMBER_DARK, lw=1.3, mutation_scale=7)
    draw_routed_arrow(ax, [(13.41, 1.64), (13.54, 1.64)], color=NAVY_PRIMARY, lw=1.3, mutation_scale=7)

    draw_tensor_slab(ax, 13.56, 1.32, 0.58, 0.58, n_slabs=3, dx=0.03, dy=0.03,
                     facecolor=NAVY_SOFT, edgecolor=NAVY_PRIMARY,
                     label='$\\mathbf{Z}_{\\mathrm{g}\\to\\mathrm{s}}$', label_fg=NAVY_PRIMARY, fontsize=8.6)

    add_pill_box(ax, 11.12, 0.56, 3.06, 0.32,
                 'Injects Evolutionary Context into 3D Nodes',
                 bg=NAVY_TINT, edge=NAVY_BORDER, fg=NAVY_PRIMARY, fontsize=8.3)

    # Symmetric internal convergence arrows inside Stage D
    draw_routed_arrow(ax, [(11.78, 6.42), (11.78, 6.02)], color=PURPLE_DARK, lw=1.9)
    ax.text(12.06, 6.22, r'$\mathbf{Z}_{\mathrm{s}\to\mathrm{g}}$', ha='left', va='center',
            fontsize=9.0, fontweight='bold', color=PURPLE_DARK)

    draw_routed_arrow(ax, [(13.52, 2.66), (13.52, 3.04)], color=NAVY_PRIMARY, lw=1.9)
    ax.text(13.24, 2.85, r'$\mathbf{Z}_{\mathrm{g}\to\mathrm{s}}$', ha='right', va='center',
            fontsize=9.0, fontweight='bold', color=NAVY_PRIMARY)

    # =========================================================================
    # STAGE E: Masked Dual Readout, Classification & Interpretability (x = 14.96 .. 18.18)
    # =========================================================================
    add_stage_container(ax, 14.96, 0.28, 3.22, 9.04, bg=CRIMSON_BG, edge=CRIMSON_BORDER)
    add_stage_header(ax, 'E', 15.12, 8.78, 2.90, 0.38, 'Readout & Interpretability', bg=CRIMSON_DARK, fontsize=10.8)

    # Card E1: Visual Masked Dual Pooling (y = 6.42 .. 8.60)
    add_card(ax, 15.12, 6.42, 2.90, 2.18, title='Masked Dual Pooling (256-D)',
             title_bg=CRIMSON_SOFT, title_fg=CRIMSON_DARK, edge=CRIMSON_BORDER, fontsize=10.0)

    add_pill_box(ax, 15.26, 7.54, 1.25, 0.56,
                 'Masked Mean\n(128-D)',
                 bg=CRIMSON_TINT, edge=CRIMSON_MAIN, fg=CRIMSON_DARK, fontsize=8.6)
    add_pill_box(ax, 16.63, 7.54, 1.25, 0.56,
                 'Masked Max\n(128-D)',
                 bg=NAVY_TINT, edge=NAVY_DEEP, fg=NAVY_PRIMARY, fontsize=8.6)

    for seg_i in range(16):
        raw_seg = plt.cm.Reds(0.45 + 0.45 * np.sin(seg_i * 1.3)**2) if seg_i < 8 else plt.cm.Blues(0.45 + 0.45 * np.cos(seg_i * 1.1)**2)
        col_seg = mute_rgba(raw_seg)
        ax.add_patch(patches.Rectangle((15.39 + seg_i * (2.36 / 16.0), 7.10),
                                       2.36 / 16.0, 0.22, facecolor=col_seg, edgecolor='white', lw=0.3, zorder=5))
    ax.add_patch(patches.Rectangle((15.39, 7.10), 2.36, 0.22, facecolor='none', edgecolor=SLATE_DARK, lw=0.9, zorder=6))
    draw_routed_arrow(ax, [(15.88, 7.54), (15.88, 7.33)], color=CRIMSON_DARK, lw=1.2, mutation_scale=7)
    draw_routed_arrow(ax, [(17.25, 7.54), (17.25, 7.33)], color=NAVY_PRIMARY, lw=1.2, mutation_scale=7)

    add_pill_box(ax, 15.26, 6.54, 2.62, 0.40,
                 'Global + Local Representation (256-D)',
                 bg='#F8FAFC', edge='#CBD5E1', fg=SLATE_TEXT, fontsize=8.2)

    # Card E2: Visual Tapered MLP Network & Softmax Output (y = 3.58 .. 6.12)
    add_card(ax, 15.12, 3.58, 2.90, 2.54, title='3-Layer MLP Classifier',
             title_bg=CRIMSON_SOFT, title_fg=CRIMSON_DARK, edge=CRIMSON_BORDER, fontsize=10.0)

    layer_y = [5.46, 5.00, 4.54, 4.08]
    layer_nodes = [
        np.linspace(15.34, 16.20, 6),
        np.linspace(15.46, 16.08, 4),
        np.linspace(15.56, 15.98, 3),
        np.linspace(15.66, 15.88, 2),
    ]
    layer_cols = [NAVY_PRIMARY, CRIMSON_MAIN, CRIMSON_MAIN, CRIMSON_DARK]
    for l_idx in range(len(layer_y) - 1):
        for x1 in layer_nodes[l_idx]:
            for x2 in layer_nodes[l_idx + 1]:
                ax.plot([x1, x2], [layer_y[l_idx], layer_y[l_idx + 1]],
                        color=CRIMSON_EDGE, lw=0.7, alpha=0.65, zorder=4)
    for l_idx, (ly_n, xs_n, col_n) in enumerate(zip(layer_y, layer_nodes, layer_cols)):
        for xn in xs_n:
            ax.scatter([xn], [ly_n], s=36, color=col_n, edgecolor='white', lw=0.6, zorder=6)

    add_pill_box(ax, 16.34, 5.31, 1.54, 0.30, '256-D Pooled',
                 bg='#F8FAFC', edge='#CBD5E1', fg=SLATE_TEXT, fontsize=8.1)
    add_pill_box(ax, 16.34, 4.85, 1.54, 0.30, '128-D (ReLU + Drop)',
                 bg=CRIMSON_TINT, edge=CRIMSON_EDGE, fg=CRIMSON_TEXT, fontsize=8.1)
    add_pill_box(ax, 16.34, 4.39, 1.54, 0.30, '32-D (ReLU + Drop)',
                 bg=CRIMSON_TINT, edge=CRIMSON_EDGE, fg=CRIMSON_TEXT, fontsize=8.1)
    add_pill_box(ax, 16.34, 3.93, 1.54, 0.30, 'Softmax Output',
                 bg=CRIMSON_SOFT, edge=CRIMSON_MAIN, fg=CRIMSON_DARK, fontsize=8.2)

    ax.text(16.57, 3.70, 'Label-Smoothed Cross-Entropy',
            ha='center', va='center', fontsize=8.2, fontweight='bold', color=SLATE_MUTED, zorder=5)

    # Card E3: Mechanistic & Biophysical Interpretability (y = 0.46 .. 3.28)
    add_card(ax, 15.12, 0.46, 2.90, 2.82, title='Mechanistic Interpretability',
             title_bg='#EEF2F6', title_fg=NAVY_PRIMARY, edge='#94A3B8', fontsize=10.0)

    px_prof = np.linspace(15.35, 17.78, 10)
    py_prof = np.array([0.20, 0.16, 0.22, 0.72, 0.28, 0.68, 0.26, 0.80, 0.92, 0.85]) * 0.84
    for x_p, h_p, aa_l in zip(px_prof, py_prof, seq_tokens):
        bcol = CRIMSON_MAIN if h_p > 0.48 else '#94A3B8'
        ax.add_patch(patches.Rectangle((x_p - 0.08, 1.78), 0.16, h_p, facecolor=bcol, alpha=0.88, zorder=5))
        ax.text(x_p, 1.60, aa_l, ha='center', va='center', fontsize=8.2, fontweight='bold', color=SLATE_TEXT, zorder=5)
    ax.plot([15.23, 17.90], [1.78, 1.78], color='#64748B', lw=0.9, zorder=5)
    ax.text(17.10, 2.74, 'C-term GPCR Core', ha='center', va='center',
            fontsize=8.2, fontweight='bold', color=CRIMSON_DARK, zorder=6)

    add_pill_box(ax, 15.26, 1.16, 2.62, 0.28,
                 'Aromatic / Basic Core (Phe, Trp, Arg)',
                 bg=CRIMSON_TINT, edge=CRIMSON_BORDER, fg=CRIMSON_DARK, fontsize=7.9)
    add_pill_box(ax, 15.26, 0.85, 2.62, 0.28,
                 'Disulfide Bridge 3D Topology',
                 bg=TEAL_TINT, edge=TEAL_BORDER, fg=TEAL_DARK, fontsize=7.9)
    add_pill_box(ax, 15.26, 0.54, 2.62, 0.28,
                 'Cleavage-Bias Shortcut Resistance',
                 bg=NAVY_TINT, edge=NAVY_BORDER, fg=NAVY_PRIMARY, fontsize=7.9)

    draw_routed_arrow(ax, [(16.57, 6.42), (16.57, 6.12)], color=CRIMSON_DARK, lw=1.9)
    draw_routed_arrow(ax, [(16.57, 3.58), (16.57, 3.28)], color=NAVY_PRIMARY, lw=1.8)

    # =========================================================================
    # PRECISION INTER-STAGE ROUTING
    # =========================================================================
    draw_routed_arrow(ax, [(3.22, 7.50), (4.06, 7.50)], color=AMBER_DARK, lw=2.1)

    draw_routed_arrow(ax, [(4.06, 6.00), (3.64, 6.00), (3.64, 3.30), (4.06, 3.30)],
                      color=AMBER_DARK, lw=1.8, ls='--')
    add_arrow_badge(ax, 3.64, 4.90, 'Contacts\n$\\mathbf{C}$', fg=AMBER_DARK, edge=AMBER_BORDER, fontsize=8.2)

    draw_routed_arrow(ax, [(3.22, 0.84), (4.06, 0.84)], color=TEAL_DARK, lw=2.1)
    add_arrow_badge(ax, 3.64, 1.20, r'$\mathbf{H}_{\mathrm{node}}$', fg=TEAL_DARK, edge=TEAL_MAIN, fontsize=8.6)

    draw_routed_arrow(ax, [(10.14, 7.95), (10.98, 7.95)], color=AMBER_DARK, lw=2.2)
    add_arrow_badge(ax, 10.56, 8.28, r'$\mathbf{H}_{\mathrm{seq}}$', fg=AMBER_DARK, edge=AMBER_BORDER, fontsize=8.8)

    draw_routed_arrow(ax, [(10.14, 1.32), (10.98, 1.32)], color=TEAL_DARK, lw=2.2)
    add_arrow_badge(ax, 10.56, 0.94, r'$\mathbf{H}_{\mathrm{graph}}$', fg=TEAL_DARK, edge=TEAL_MAIN, fontsize=8.8)

    ax.scatter([10.42], [7.95], s=28, color=AMBER_DARK, edgecolor='white', lw=0.6, zorder=9)
    ax.scatter([10.42], [3.65], s=24, color=AMBER_DARK, edgecolor='white', lw=0.5, zorder=9)
    draw_routed_arrow(ax, [(10.42, 7.95), (10.42, 3.65), (10.14, 3.65)],
                      color=AMBER_DARK, lw=1.6, ls='--', mutation_scale=9, zorder=8)
    add_arrow_badge(ax, 10.42, 5.26, 'Init\n$\\mathbf{H}_{\\mathrm{seq}}$', fg=AMBER_DARK, edge=AMBER_BORDER, fontsize=8.0)
    draw_routed_arrow(ax, [(10.42, 3.65), (10.42, 1.61), (10.98, 1.61)],
                      color=AMBER_DARK, lw=1.35, ls='--', alpha=0.82, mutation_scale=8, zorder=6)

    ax.scatter([10.68], [1.32], s=28, color=TEAL_DARK, edgecolor='white', lw=0.6, zorder=9)
    ax.plot([10.68, 10.68], [1.54, 1.68], color='white', lw=4.2, zorder=6.5)
    draw_routed_arrow(ax, [(10.68, 1.32), (10.68, 7.42), (10.98, 7.42)],
                      color=TEAL_DARK, lw=1.35, ls='--', alpha=0.82, mutation_scale=8, zorder=7)

    draw_routed_arrow(ax, [(14.32, 4.05), (14.72, 4.05), (14.72, 7.52), (15.12, 7.52)],
                      color=INDIGO_DARK, lw=2.2)
    add_arrow_badge(ax, 14.72, 5.85, r'$\mathbf{H}_{\mathrm{fuse}}$', fg=INDIGO_DARK, edge=INDIGO_BORDER, fontsize=8.8)
    ax.scatter([14.72], [4.05], s=30, color=INDIGO_DARK, edgecolor='white', lw=0.6, zorder=9)

    draw_routed_arrow(ax, [(14.72, 4.05), (14.72, 2.05), (15.12, 2.05)],
                      color=PURPLE_DARK, lw=1.4, ls='--', mutation_scale=9, zorder=7)
    add_arrow_badge(ax, 14.72, 2.95, 'Attn\nMaps', fg=PURPLE_DARK, edge=PURPLE_BORDER, fontsize=8.0)

    plt.tight_layout(pad=0.10)
    pdf_out = os.path.join(PAPER_FIG_DIR, 'fig0_architecture.pdf')
    png_out = os.path.join(PAPER_FIG_DIR, 'fig0_architecture.png')
    plt.savefig(pdf_out, bbox_inches='tight')
    plt.savefig(png_out, dpi=350, bbox_inches='tight')
    plt.close()
    print(f'Successfully generated large-font visual {pdf_out} and {png_out}')


if __name__ == '__main__':
    build_architecture_figure()
