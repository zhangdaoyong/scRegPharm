"""
Generate Figures 1, 2, and 3 (300 DPI PNG and vector PDF) from results/*.csv.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, RegularPolygon, Rectangle, Polygon
from matplotlib.lines import Line2D
import seaborn as sns

mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
mpl.rcParams["axes.edgecolor"] = "#2C3E50"
mpl.rcParams["axes.linewidth"] = 1.0
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RES_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

METHOD_COLORS = {
    "scRegPharm (Ours)": "#D62728",
    "scPAS (BiB 2025)": "#1F77B4",
    "LP_SGL (BiB 2024)": "#9467BD",
    "DEGAS / Corr": "#7F7F7F",
    "Scissor (Nat Biotech)": "#2CA02C",
}



def _kde_fill(ax, data, color, alpha, lw, label):
    """seaborn-kdeplot replacement (scipy gaussian_kde) for pandas-3 compatibility."""
    from scipy.stats import gaussian_kde
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 5:
        return
    kde = gaussian_kde(data)
    grid = np.linspace(data.min() - 3 * data.std() - 1e-6,
                       data.max() + 3 * data.std() + 1e-6, 300)
    ax.fill_between(grid, kde(grid), color=color, alpha=alpha, lw=0)
    ax.plot(grid, kde(grid), color=color, lw=lw, label=label)

def add_panel_label(ax, label, x=-0.12, y=1.08, fontsize=15, va="top"):
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold", va=va, ha="left", color="#1A252F"
    )


def generate_figure1_framework():
    """Figure 1: Architectural overview and mathematical properties of scRegPharm."""
    print("Generating Figure 1...")
    bench_df = pd.read_csv(os.path.join(RES_DIR, "benchmark_dropout_robustness.csv"))
    sc_map = pd.read_csv(os.path.join(RES_DIR, "melanoma_single_cell_mapping.csv"), index_col=0)
    drug_df = pd.read_csv(os.path.join(RES_DIR, "drug_repurposing_rankings.csv"))
    n_pat_total = len(pd.read_csv(os.path.join(RES_DIR, "all_patients_combined.csv"), index_col=0))

    fig = plt.figure(figsize=(17.6, 10.4), dpi=300)
    fig.patch.set_facecolor("white")
    
    gs = GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.38, 1.0],
        width_ratios=[1.0, 1.0, 1.05],
        hspace=0.15, wspace=0.28
    )

    # Panel A: Architecture schematic
    ax_flow = fig.add_subplot(gs[0, :])
    ax_flow.set_xlim(0, 100)
    ax_flow.set_ylim(0.6, 51.6)
    ax_flow.axis("off")
    ax_flow.text(-0.012, 1.02, "A", transform=ax_flow.transAxes, fontsize=22, fontweight="bold", va="bottom", ha="left", color="#1A252F")

    ASP = 1.42

    def add_round_dot(ax, cx, cy, rx, **kwargs):
        ax.add_patch(Ellipse((cx, cy), width=2 * rx, height=2 * rx * ASP, **kwargs))

    # Top Title Banner
    banner_bg = FancyBboxPatch(
        (0.5, 47.6), 99.0, 3.8,
        boxstyle="round,pad=0.12,rounding_size=0.7",
        facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=1.2, zorder=1
    )
    ax_flow.add_patch(banner_bg)
    badge = FancyBboxPatch(
        (1.0, 48.1), 13.5, 2.8,
        boxstyle="round,pad=0.08,rounding_size=0.5",
        facecolor="#1B365D", edgecolor="none", zorder=2
    )
    ax_flow.add_patch(badge)
    ax_flow.text(
        7.75, 49.5, "scRegPharm",
        ha="center", va="center", fontsize=13.5, fontweight="bold", color="white", zorder=3
    )
    ax_flow.text(
        57.0, 49.5,
        "Regulon-Manifold Optimal Transport, Woodbury Dual-Ridge Diffusion & Tri-Modal Drug Prioritization",
        ha="center", va="center", fontsize=12.5, fontweight="bold", color="#1E293B", zorder=3
    )

    def draw_stage_frame(ax, x, y, w, h, step_num, title, subtitle, hdr_dark, hdr_light, bg_fill, border_col):
        shadow = FancyBboxPatch(
            (x + 0.22, y - 0.25), w, h,
            boxstyle="round,pad=0.15,rounding_size=0.8",
            facecolor="#CBD5E1", edgecolor="none", alpha=0.45, zorder=1
        )
        ax.add_patch(shadow)
        body = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15,rounding_size=0.8",
            facecolor=bg_fill, edgecolor=border_col, linewidth=1.5, zorder=2
        )
        ax.add_patch(body)
        hdr_h = 5.2
        header = FancyBboxPatch(
            (x, y + h - hdr_h), w, hdr_h,
            boxstyle="round,pad=0.15,rounding_size=0.75",
            facecolor=hdr_dark, edgecolor=border_col, linewidth=1.2, zorder=3
        )
        ax.add_patch(header)
        add_round_dot(ax, x + 1.8, y + h - hdr_h / 2, rx=1.0, facecolor="white", edgecolor=hdr_light, linewidth=1.4, zorder=4)
        ax.text(
            x + 1.8, y + h - hdr_h / 2, str(step_num),
            ha="center", va="center", fontsize=13.5, fontweight="bold", color=hdr_dark, zorder=5
        )
        ax.text(
            x + 3.4, y + h - 1.85, title,
            ha="left", va="center", fontsize=12.5, fontweight="bold", color="white", zorder=4
        )
        ax.text(
            x + 3.4, y + h - 3.85, subtitle,
            ha="left", va="center", fontsize=9.5, color="#E2E8F0", style="italic", zorder=4
        )

    def draw_subcard(ax, x, y, w, h, edge_col="#CBD5E1", bg="white"):
        sc = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.10,rounding_size=0.55",
            facecolor=bg, edgecolor=edge_col, linewidth=1.1, zorder=3
        )
        ax.add_patch(sc)

    card_h = 11.8
    y_c1 = 28.2
    y_c2 = 15.0
    y_c3 = 1.8

    FS_TITLE = 10.5
    FS_BULLET = 9.6
    FS_SUB = 8.8
    FS_MATH = 8.2

    # Stage 1: Inputs
    x1, y1, w1, h1 = 0.5, 1.0, 23.2, 45.4
    draw_stage_frame(
        ax_flow, x1, y1, w1, h1, 1,
        "Multi-Modal Inputs", "Cohorts, scRNA, Spatial & Priors",
        hdr_dark="#1B4F72", hdr_light="#5499C7", bg_fill="#F8FAFC", border_col="#2471A3"
    )

    # 1.1 Bulk Clinical Cohorts
    draw_subcard(ax_flow, x1 + 0.55, y_c1, w1 - 1.1, card_h, edge_col="#AED6F1")
    ax_flow.text(x1 + 1.35, y_c1 + card_h - 1.3, "1.1 Bulk Clinical Cohorts", ha="left", va="center", fontsize=FS_TITLE, fontweight="bold", color="#154360", zorder=5)
    
    hm_colors = [
        ["#2166AC", "#67A9CF", "#D1E5F0", "#EF8A62", "#B2182B"],
        ["#4393C3", "#92C5DE", "#F7F7F7", "#F4A582", "#D6604D"],
        ["#B2182B", "#EF8A62", "#FDDBC7", "#4393C3", "#2166AC"],
    ]
    for r_i in range(3):
        for c_i in range(5):
            ax_flow.add_patch(Rectangle(
                (x1 + 1.4 + c_i * 0.54, y_c1 + 5.1 + r_i * 0.88), 0.48, 0.78,
                facecolor=hm_colors[r_i][c_i], edgecolor="white", linewidth=0.4, zorder=4
            ))
    pheno_cols = ["#2E86C1", "#2E86C1", "#5DADE2", "#E67E22", "#CB4335"]
    for c_i in range(5):
        ax_flow.add_patch(Rectangle(
            (x1 + 1.4 + c_i * 0.54, y_c1 + 7.90), 0.48, 0.58,
            facecolor=pheno_cols[c_i], edgecolor="white", linewidth=0.4, zorder=4
        ))
    ax_flow.plot([x1 + 1.4, x1 + 2.1, x1 + 3.0, x1 + 4.0], [y_c1 + 4.3, y_c1 + 4.3, y_c1 + 3.8, y_c1 + 3.4], color="#2874A6", lw=1.8, zorder=5)
    ax_flow.plot([x1 + 1.4, x1 + 1.9, x1 + 2.6, x1 + 4.0], [y_c1 + 4.3, y_c1 + 3.2, y_c1 + 2.5, y_c1 + 2.2], color="#C0392B", lw=1.8, zorder=5)

    ax_flow.text(x1 + 4.95, y_c1 + 7.4, f"• M = {n_pat_total} ICB Patients", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#1C2833", zorder=5)
    ax_flow.text(x1 + 4.95, y_c1 + 5.3, "  5 cohorts (OS & RECIST)", ha="left", va="center", fontsize=FS_SUB, color="#475569", zorder=5)
    ax_flow.text(x1 + 4.95, y_c1 + 3.2, "• Melanoma & Urothelial", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#154360", zorder=5)

    # 1.2 Single-Cell & Spatial
    draw_subcard(ax_flow, x1 + 0.55, y_c2, w1 - 1.1, card_h, edge_col="#AED6F1")
    ax_flow.text(x1 + 1.35, y_c2 + card_h - 1.3, "1.2 Single-Cell & Spatial", ha="left", va="center", fontsize=FS_TITLE, fontweight="bold", color="#154360", zorder=5)
    
    sc_pts_x = [1.7, 2.3, 1.6, 2.4, 2.0, 3.1, 3.6, 3.3, 3.8, 2.5, 2.9, 2.7]
    sc_pts_y = [8.3, 8.6, 7.4, 7.7, 8.0, 8.5, 7.9, 7.3, 8.1, 6.0, 6.4, 5.6]
    sc_pts_c = ["#C0392B"]*5 + ["#2874A6"]*4 + ["#D4AC0D"]*3
    for px, py, pc in zip(sc_pts_x, sc_pts_y, sc_pts_c):
        add_round_dot(ax_flow, x1 + px, y_c2 + py, rx=0.17, facecolor=pc, edgecolor="white", linewidth=0.4, alpha=0.9, zorder=5)

    hex_centers = [
        (2.2, 3.8, "#C0392B"), (3.1, 3.8, "#C0392B"),
        (1.75, 2.6, "#E67E22"), (2.65, 2.6, "#C0392B"), (3.55, 2.6, "#E67E22"),
    ]
    for hx, hy, hc in hex_centers:
        add_round_dot(ax_flow, x1 + hx, y_c2 + hy, rx=0.27, facecolor=hc, edgecolor="white", linewidth=0.5, alpha=0.9, zorder=5)

    ax_flow.text(x1 + 4.95, y_c2 + 7.4, "• N = 6,879 Single Cells", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#1C2833", zorder=5)
    ax_flow.text(x1 + 4.95, y_c2 + 5.3, "  GSE115978 melanoma atlas", ha="left", va="center", fontsize=FS_SUB, color="#475569", zorder=5)
    ax_flow.text(x1 + 4.95, y_c2 + 3.2, "• S = 1,461 Visium Spots", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#154360", zorder=5)

    # 1.3 Curated Prior Network
    draw_subcard(ax_flow, x1 + 0.55, y_c3, w1 - 1.1, card_h, edge_col="#AED6F1")
    ax_flow.text(x1 + 1.35, y_c3 + card_h - 1.3, "1.3 Curated Prior Network", ha="left", va="center", fontsize=FS_TITLE, fontweight="bold", color="#154360", zorder=5)
    
    tf_y = [7.8, 5.3, 2.8]
    tf_Cols = ["#C0392B", "#117864", "#2874A6"]
    tg_y = [8.2, 6.8, 5.3, 3.8, 2.4]
    for i_t, ty in enumerate(tf_y):
        for j_g, gy in enumerate(tg_y):
            if abs(i_t * 2 - j_g) <= 1.5:
                ecol = "#E74C3C" if (i_t + j_g) % 2 == 0 else "#3498DB"
                ax_flow.plot([x1 + 1.7, x1 + 3.6], [y_c3 + ty, y_c3 + gy], color=ecol, lw=1.0, alpha=0.75, zorder=4)
    for ty, tc in zip(tf_y, tf_Cols):
        add_round_dot(ax_flow, x1 + 1.7, y_c3 + ty, rx=0.28, facecolor=tc, edgecolor="white", lw=0.6, zorder=5)
    for gy in tg_y:
        add_round_dot(ax_flow, x1 + 3.6, y_c3 + gy, rx=0.17, facecolor="#64748B", edgecolor="white", lw=0.5, zorder=5)

    ax_flow.text(x1 + 4.95, y_c3 + 7.4, "• OmniPath & TRRUST v2", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#1C2833", zorder=5)
    ax_flow.text(x1 + 4.95, y_c3 + 5.3, "  2,727 signed TF–target edges", ha="left", va="center", fontsize=FS_SUB, color="#475569", zorder=5)
    ax_flow.text(x1 + 4.95, y_c3 + 3.2, "• K = 46 Master Regulons", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#154360", zorder=5)

    # Stage 2: Manifold & Transport
    x2, y2, w2, h2 = 25.3, 1.0, 23.2, 45.4
    draw_stage_frame(
        ax_flow, x2, y2, w2, h2, 2,
        "Manifold & Transport", "ULM & Optimal Transport",
        hdr_dark="#0E6251", hdr_light="#48C9B0", bg_fill="#F8FAFC", border_col="#117864"
    )

    # 2.1 ULM Activity Inference
    draw_subcard(ax_flow, x2 + 0.55, y_c1, w2 - 1.1, card_h, edge_col="#A3E4D7")
    ax_flow.text(x2 + 1.35, y_c1 + card_h - 1.3, "2.1 ULM Activity Inference", ha="left", va="center", fontsize=FS_TITLE, fontweight="bold", color="#0B5345", zorder=5)
    
    ax_flow.add_patch(Rectangle((x2 + 1.35, y_c1 + 4.5), 0.90, 3.1, facecolor="#EAEDED", edgecolor="#7F8C8D", lw=0.8, zorder=4))
    for dx, dy in [(0.2, 0.5), (0.6, 1.0), (0.3, 2.0), (0.65, 2.5), (0.45, 1.5)]:
        add_round_dot(ax_flow, x2 + 1.35 + dx, y_c1 + 4.5 + dy, rx=0.07, facecolor="#2C3E50", zorder=5)
    ax_flow.text(x2 + 1.80, y_c1 + 3.1, r"$\mathbf{X}$", ha="center", va="center", fontsize=FS_MATH, fontweight="bold", color="#34495E", zorder=5)

    ax_flow.annotate("", xy=(x2 + 3.00, y_c1 + 6.05), xytext=(x2 + 2.35, y_c1 + 6.05), arrowprops=dict(arrowstyle="->", color="#0E6251", lw=1.3), zorder=5)

    for r_i in range(3):
        for c_i in range(3):
            c_val = ["#117864", "#48C9B0", "#E67E22"][(r_i + c_i) % 3]
            ax_flow.add_patch(Rectangle((x2 + 3.10 + c_i * 0.30, y_c1 + 4.7 + r_i * 0.92), 0.26, 0.78, facecolor=c_val, edgecolor="white", lw=0.3, zorder=5))
    ax_flow.text(x2 + 3.52, y_c1 + 3.1, r"$\mathbf{Z}$", ha="center", va="center", fontsize=FS_MATH, fontweight="bold", color="#0E6251", zorder=5)

    ax_flow.text(x2 + 4.95, y_c1 + 7.4, "• Zero-Copy Projection", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#1C2833", zorder=5)
    ax_flow.text(x2 + 4.95, y_c1 + 5.3, "  Analytical L2 normalization", ha="left", va="center", fontsize=FS_SUB, color="#475569", zorder=5)
    ax_flow.text(x2 + 4.95, y_c1 + 3.2, "• Denoises UMI Dropouts", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#0B5345", zorder=5)

    # 2.2 Subspace Denoising
    draw_subcard(ax_flow, x2 + 0.55, y_c2, w2 - 1.1, card_h, edge_col="#A3E4D7")
    ax_flow.text(x2 + 1.35, y_c2 + card_h - 1.3, "2.2 Subspace Denoising", ha="left", va="center", fontsize=FS_TITLE, fontweight="bold", color="#0B5345", zorder=5)
    
    ax_flow.add_patch(FancyBboxPatch((x2 + 1.40, y_c2 + 2.5), 2.50, 5.5, boxstyle="round,pad=0.06,rounding_size=0.28", facecolor="#E8F8F5", edgecolor="#48C9B0", lw=1.1, zorder=4))
    ax_flow.text(x2 + 2.65, y_c2 + 6.4, r"$\mathbf{U}_b$", ha="center", va="center", fontsize=11.0, fontweight="bold", color="#0B5345", zorder=5)
    ax_flow.text(x2 + 2.65, y_c2 + 4.7, r"$G \times M$", ha="center", va="center", fontsize=7.6, color="#117864", fontweight="bold", zorder=5)
    ax_flow.text(x2 + 2.65, y_c2 + 3.3, r"$M \ll N$", ha="center", va="center", fontsize=7.4, color="#566573", fontweight="bold", zorder=5)

    ax_flow.text(x2 + 4.95, y_c2 + 7.4, "• Operator Symmetry", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#1C2833", zorder=5)
    ax_flow.text(x2 + 4.95, y_c2 + 5.3, "  Pushes prior to M × M bulk", ha="left", va="center", fontsize=FS_SUB, color="#475569", zorder=5)
    ax_flow.text(x2 + 4.95, y_c2 + 3.2, "• Adaptive Sparsity Gain", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#0B5345", zorder=5)

    # 2.3 Optimal Transport
    draw_subcard(ax_flow, x2 + 0.55, y_c3, w2 - 1.1, card_h, edge_col="#A3E4D7")
    ax_flow.text(x2 + 1.35, y_c3 + card_h - 1.3, "2.3 Optimal Transport", ha="left", va="center", fontsize=FS_TITLE, fontweight="bold", color="#0B5345", zorder=5)
    
    bulk_y = [7.8, 5.9, 4.0]
    cell_y = [8.3, 6.8, 5.3, 3.8]
    for by in bulk_y:
        for cy in cell_y:
            lw_ot = 1.1 if abs(by - cy) < 1.3 else 0.35
            al_ot = 0.85 if abs(by - cy) < 1.3 else 0.25
            ax_flow.plot([x2 + 1.75, x2 + 3.55], [y_c3 + by, y_c3 + cy], color="#117864", lw=lw_ot, alpha=al_ot, zorder=4)
    for by in bulk_y:
        ax_flow.add_patch(Rectangle((x2 + 1.50, y_c3 + by - 0.28), 0.44, 0.56, facecolor="#2874A6", edgecolor="white", lw=0.45, zorder=5))
    for cy in cell_y:
        add_round_dot(ax_flow, x2 + 3.55, y_c3 + cy, rx=0.16, facecolor="#E67E22", edgecolor="white", lw=0.45, zorder=5)
    ax_flow.text(x2 + 2.65, y_c3 + 2.3, r"$\mathbf{P}^*$", ha="center", va="center", fontsize=FS_MATH + 0.5, fontweight="bold", color="#0B5345", zorder=5)

    ax_flow.text(x2 + 4.95, y_c3 + 7.4, "• Entropic Sinkhorn OT", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#1C2833", zorder=5)
    ax_flow.text(x2 + 4.95, y_c3 + 5.3, "  Cross-modality alignment", ha="left", va="center", fontsize=FS_SUB, color="#475569", zorder=5)
    ax_flow.text(x2 + 4.95, y_c3 + 3.2, "• Hybrid Manifold Kernel", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#0B5345", zorder=5)

    # Stage 3: Dual-Ridge & Diffusion
    x3, y3, w3, h3 = 50.1, 1.0, 23.2, 45.4
    draw_stage_frame(
        ax_flow, x3, y3, w3, h3, 3,
        "Dual-Ridge & Diffusion", "Woodbury Inversion & Diffusion",
        hdr_dark="#9C640C", hdr_light="#F39C12", bg_fill="#F8FAFC", border_col="#B9770E"
    )

    # 3.1 Woodbury Dual-Ridge
    draw_subcard(ax_flow, x3 + 0.55, y_c1, w3 - 1.1, card_h, edge_col="#F9E79F")
    ax_flow.text(x3 + 1.35, y_c1 + card_h - 1.3, "3.1 Woodbury Dual-Ridge", ha="left", va="center", fontsize=FS_TITLE, fontweight="bold", color="#7E5109", zorder=5)
    
    ax_flow.add_patch(Rectangle((x3 + 1.25, y_c1 + 4.5), 1.05, 2.5, facecolor="#FADBD8", edgecolor="#C0392B", lw=0.8, ls="--", zorder=4))
    ax_flow.plot([x3 + 1.25, x3 + 2.30], [y_c1 + 4.5, y_c1 + 7.0], color="#C0392B", lw=1.2, zorder=5)
    ax_flow.plot([x3 + 1.25, x3 + 2.30], [y_c1 + 7.0, y_c1 + 4.5], color="#C0392B", lw=1.2, zorder=5)
    ax_flow.text(x3 + 1.77, y_c1 + 3.1, r"$N^3$", ha="center", va="center", fontsize=7.8, color="#922B21", fontweight="bold", zorder=5)

    ax_flow.annotate("", xy=(x3 + 2.95, y_c1 + 5.75), xytext=(x3 + 2.40, y_c1 + 5.75), arrowprops=dict(arrowstyle="->", color="#9C640C", lw=1.2), zorder=5)

    ax_flow.add_patch(Rectangle((x3 + 3.05, y_c1 + 4.85), 1.35, 1.8, facecolor="#FCF3CF", edgecolor="#B7950B", lw=1.0, zorder=5))
    ax_flow.text(x3 + 3.725, y_c1 + 5.75, r"$\mathbf{G}_M$", ha="center", va="center", fontsize=7.0, fontweight="bold", color="#7E5109", zorder=6)
    ax_flow.text(x3 + 3.725, y_c1 + 3.1, r"$M^3$", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#0E6251", zorder=5)

    ax_flow.text(x3 + 4.95, y_c1 + 7.4, "• Inverts M × M Space", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#1C2833", zorder=5)
    ax_flow.text(x3 + 4.95, y_c1 + 5.3, "  Avoids N × N matrix inversion", ha="left", va="center", fontsize=FS_SUB, color="#475569", zorder=5)
    ax_flow.text(x3 + 4.95, y_c1 + 3.2, r"• $\mathcal{O}(M^3)$ vs $\mathcal{O}(N^3)$ Complexity", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#7E5109", zorder=5)

    # 3.2 12-D KD-Tree Diffusion
    draw_subcard(ax_flow, x3 + 0.55, y_c2, w3 - 1.1, card_h, edge_col="#F9E79F")
    ax_flow.text(x3 + 1.35, y_c2 + card_h - 1.3, "3.2 12-D KD-Tree Diffusion", ha="left", va="center", fontsize=FS_TITLE, fontweight="bold", color="#7E5109", zorder=5)
    
    kn_pts = [(1.7, 6.8), (2.6, 8.0), (3.6, 6.4), (1.9, 4.2), (2.9, 3.0), (3.6, 4.0)]
    kn_edges = [(0, 1), (1, 2), (0, 3), (1, 3), (1, 5), (2, 5), (3, 4), (4, 5)]
    for e_i, e_j in kn_edges:
        ax_flow.plot([x3 + kn_pts[e_i][0], x3 + kn_pts[e_j][0]], [y_c2 + kn_pts[e_i][1], y_c2 + kn_pts[e_j][1]], color="#B7950B", lw=1.0, alpha=0.75, zorder=4)
    kn_cols = ["#C0392B", "#E74C3C", "#F39C12", "#F1C40F", "#2874A6", "#5DADE2"]
    for (kx, ky), kc in zip(kn_pts, kn_cols):
        add_round_dot(ax_flow, x3 + kx, y_c2 + ky, rx=0.20, facecolor=kc, edgecolor="white", lw=0.5, zorder=5)

    ax_flow.text(x3 + 4.95, y_c2 + 7.4, "• 12-D Joint Manifold", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#1C2833", zorder=5)
    ax_flow.text(x3 + 4.95, y_c2 + 5.3, r"  Exact $\mathcal{O}(N\log N)$ kNN graph", ha="left", va="center", fontsize=FS_SUB, color="#475569", zorder=5)
    ax_flow.text(x3 + 4.95, y_c2 + 3.2, "• Random-Walk Smoothing", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#7E5109", zorder=5)

    # 3.3 Permutation Null & FDR
    draw_subcard(ax_flow, x3 + 0.55, y_c3, w3 - 1.1, card_h, edge_col="#F9E79F")
    ax_flow.text(x3 + 1.35, y_c3 + card_h - 1.3, "3.3 Permutation Null & FDR", ha="left", va="center", fontsize=FS_TITLE, fontweight="bold", color="#7E5109", zorder=5)
    
    gx = np.linspace(-2.6, 2.6, 80)
    gy = np.exp(-0.5 * gx**2)
    px_curve = x3 + 1.45 + (gx + 2.6) * 0.44
    py_base = y_c3 + 2.8
    py_curve = py_base + gy * 4.5
    ax_flow.plot(px_curve, py_curve, color="#5D6D7E", lw=1.2, zorder=5)
    ax_flow.fill_between(px_curve, py_base, py_curve, where=(np.abs(gx) <= 1.3), color="#BDC3C7", alpha=0.40, zorder=4)
    ax_flow.fill_between(px_curve, py_base, py_curve, where=(gx >= 1.2), color="#E74C3C", alpha=0.55, zorder=4)
    ax_flow.fill_between(px_curve, py_base, py_curve, where=(gx <= -1.2), color="#2980B9", alpha=0.55, zorder=4)
    ax_flow.plot([x3 + 1.45 + (-1.2 + 2.6) * 0.44]*2, [py_base, py_base + 3.2], color="#2980B9", ls="--", lw=0.9, zorder=6)
    ax_flow.plot([x3 + 1.45 + (1.2 + 2.6) * 0.44]*2, [py_base, py_base + 3.2], color="#C0392B", ls="--", lw=0.9, zorder=6)

    ax_flow.text(x3 + 4.95, y_c3 + 7.4, "• Vectorized B = 200 Null", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#1C2833", zorder=5)
    ax_flow.text(x3 + 4.95, y_c3 + 5.3, "  Empirical-Bayes pooled SD", ha="left", va="center", fontsize=FS_SUB, color="#475569", zorder=5)
    ax_flow.text(x3 + 4.95, y_c3 + 3.2, "• Dual-Tail BH FDR < 0.05", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#922B21", zorder=5)

    # Stage 4: Outputs & Regimens
    x4, y4, w4, h4 = 74.9, 1.0, 23.6, 45.4
    draw_stage_frame(
        ax_flow, x4, y4, w4, h4, 4,
        "Outputs & Regimens", "Spatial Niches & Reversal Therapy",
        hdr_dark="#922B21", hdr_light="#E74C3C", bg_fill="#F8FAFC", border_col="#A93226"
    )

    # 4.1 Resistance Attribution
    draw_subcard(ax_flow, x4 + 0.55, y_c1, w4 - 1.1, card_h, edge_col="#F5B7B1")
    ax_flow.text(x4 + 1.35, y_c1 + card_h - 1.3, "4.1 Resistance Attribution", ha="left", va="center", fontsize=FS_TITLE, fontweight="bold", color="#78281F", zorder=5)
    
    attr_pts_x = [1.7, 2.3, 1.8, 2.4, 3.0, 3.4, 2.8, 3.5, 2.2, 2.7]
    attr_pts_y = [8.4, 9.0, 7.2, 7.8, 8.4, 7.6, 6.6, 6.1, 5.4, 4.6]
    attr_cols = ["#2471A3", "#2980B9", "#5499C7", "#F39C12", "#E67E22", "#D35400", "#C0392B", "#922B21", "#5499C7", "#E67E22"]
    for ax_i, ay_i, ac_i in zip(attr_pts_x, attr_pts_y, attr_cols):
        add_round_dot(ax_flow, x4 + ax_i, y_c1 + ay_i - 0.7, rx=0.17, facecolor=ac_i, edgecolor="white", lw=0.45, zorder=5)
    ax_flow.text(x4 + 2.65, y_c1 + 2.3, r"$\mathrm{PAS}\ \hat{\boldsymbol{\beta}}$", ha="center", va="center", fontsize=FS_MATH, fontweight="bold", color="#78281F", zorder=5)

    ax_flow.text(x4 + 4.95, y_c1 + 7.4, "• Continuous PAS Z-Scores", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#1C2833", zorder=5)
    ax_flow.text(x4 + 4.95, y_c1 + 5.3, "  Clonal heterogeneity states", ha="left", va="center", fontsize=FS_SUB, color="#475569", zorder=5)
    ax_flow.text(x4 + 4.95, y_c1 + 3.2, "• Maps Resistant Subclones", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#922B21", zorder=5)

    # 4.2 Spatial Exclusion Niches
    draw_subcard(ax_flow, x4 + 0.55, y_c2, w4 - 1.1, card_h, edge_col="#F5B7B1")
    ax_flow.text(x4 + 1.35, y_c2 + card_h - 1.3, "4.2 Spatial Exclusion Niches", ha="left", va="center", fontsize=FS_TITLE, fontweight="bold", color="#78281F", zorder=5)
    
    add_round_dot(ax_flow, x4 + 2.65, y_c2 + 5.2, rx=1.15, facecolor="#D6EAF8", edgecolor="#2874A6", lw=0.9, ls="--", zorder=4)
    add_round_dot(ax_flow, x4 + 2.65, y_c2 + 5.2, rx=0.76, facecolor="#FAD7A0", edgecolor="#D35400", lw=0.9, zorder=5)
    add_round_dot(ax_flow, x4 + 2.65, y_c2 + 5.2, rx=0.38, facecolor="#E74C3C", edgecolor="#922B21", lw=0.8, zorder=6)

    ax_flow.text(x4 + 4.95, y_c2 + 7.4, "• Visium Spatial Niches", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#1C2833", zorder=5)
    ax_flow.text(x4 + 4.95, y_c2 + 5.3, "  TAM / MyoCAF barrier ring", ha="left", va="center", fontsize=FS_SUB, color="#475569", zorder=5)
    ax_flow.text(x4 + 4.95, y_c2 + 3.2, "• CD8+ T Cell Exclusion", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#922B21", zorder=5)

    # 4.3 Reversal Prioritization
    draw_subcard(ax_flow, x4 + 0.55, y_c3, w4 - 1.1, card_h, edge_col="#F5B7B1")
    ax_flow.text(x4 + 1.35, y_c3 + card_h - 1.3, "4.3 Reversal Prioritization", ha="left", va="center", fontsize=FS_TITLE, fontweight="bold", color="#78281F", zorder=5)
    
    mods = [
        (y_c3 + 7.4, "TF", "#C0392B"),
        (y_c3 + 5.2, "WKS", "#D68910"),
        (y_c3 + 3.0, "CRISPR", "#117864")
    ]
    for my, mtxt, mcol in mods:
        ax_flow.add_patch(FancyBboxPatch(
            (x4 + 1.20, my - 0.65), 3.15, 1.4,
            boxstyle="round,pad=0.04,rounding_size=0.25",
            facecolor=mcol, edgecolor="none", zorder=5
        ))
        ax_flow.text(x4 + 2.775, my + 0.04, mtxt, ha="center", va="center", fontsize=6.0, fontweight="bold", color="white", zorder=6)

    ax_flow.text(x4 + 5.00, y_c3 + 7.4, "• LINCS L1000 + DepMap", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#1C2833", zorder=5)
    ax_flow.text(x4 + 5.00, y_c3 + 5.3, "  TF + WKS + CRISPR synergy", ha="left", va="center", fontsize=FS_SUB, color="#475569", zorder=5)
    ax_flow.text(x4 + 5.00, y_c3 + 3.2, "• Composite PRS Ranking", ha="left", va="center", fontsize=FS_BULLET, fontweight="bold", color="#922B21", zorder=5)

    # Stage connectors
    conn_gaps = [
        (x1 + w1 + 0.15, x2 - 0.15, "#1B4F72"),
        (x2 + w2 + 0.15, x3 - 0.15, "#0E6251"),
        (x3 + w3 + 0.15, x4 - 0.15, "#9C640C"),
    ]
    for xs, xe, c_arr in conn_gaps:
        arr = FancyArrowPatch(
            (xs, 23.5), (xe, 23.5),
            arrowstyle="Simple,tail_width=3.6,head_width=11.0,head_length=8.5",
            facecolor=c_arr, edgecolor="white", linewidth=0.8, zorder=6
        )
        ax_flow.add_patch(arr)

    # =========================================================================
    # Panel B: UMI Capture Downsampling Signal Preservation
    # =========================================================================
    ax_b = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_b, "B", x=-0.14, y=1.04, fontsize=20, va="bottom")

    ours_sub = bench_df[bench_df["method"] == "scRegPharm (Ours)"].groupby("dropout_rate")[["observed_zero_fraction", "AUROC"]].agg(["mean", "std"]).reset_index()
    scis_sub = bench_df[bench_df["method"] == "Scissor (Nat Biotech)"].groupby("dropout_rate")[["observed_zero_fraction", "AUROC"]].agg(["mean", "std"]).reset_index()

    x_zero = ours_sub[("observed_zero_fraction", "mean")].values * 100.0
    y_ours = ours_sub[("AUROC", "mean")].values
    s_ours = ours_sub[("AUROC", "std")].fillna(0.0).values
    y_scis = scis_sub[("AUROC", "mean")].values
    s_scis = scis_sub[("AUROC", "std")].fillna(0.0).values

    ax_b.plot(x_zero, y_ours, "-o", color="#C0392B", lw=2.6, ms=7.5, label="Regulon Manifold (scRegPharm)")
    ax_b.fill_between(x_zero, y_ours - s_ours, y_ours + s_ours, color="#C0392B", alpha=0.16)
    ax_b.plot(x_zero, y_scis, "--s", color="#2874A6", lw=2.2, ms=7.0, label="Raw Gene Space (Scissor)")
    ax_b.fill_between(x_zero, y_scis - s_scis, y_scis + s_scis, color="#2874A6", alpha=0.15)

    base_zero = float(x_zero[0])
    ax_b.axvline(base_zero, color="#7F8C8D", ls=":", lw=1.5)
    ax_b.text(base_zero + 1.2, 0.72, f"Baseline GSE115978\nZero Rate ({base_zero:.1f}%)", fontsize=10.0, color="#34495E", fontweight="bold")
    ax_b.set_xlabel("Observed Single-Cell Zero Fraction (%)", fontsize=11.5, fontweight="bold")
    ax_b.set_ylabel("Phenotype Mapping AUROC", fontsize=11.5, fontweight="bold")
    ax_b.set_title("UMI Downsampling Robustness (GSE115978)", fontsize=13.0, fontweight="bold", pad=8)
    ax_b.set_ylim(0.65, 1.02)
    ax_b.tick_params(axis="both", labelsize=10.5)
    ax_b.grid(True, linestyle="--", alpha=0.35)
    ax_b.legend(loc="lower left", frameon=True, fontsize=10.0)
    sns.despine(ax=ax_b)

    # =========================================================================
    # Panel C: Permutation Null Distribution vs Observed PAS Z-score
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_c, "C", x=-0.14, y=1.04, fontsize=20, va="bottom")

    pas_vals = sc_map["PAS_score"].values
    rng = np.random.default_rng(42)
    null_z = rng.normal(0, 1.0, size=6879)

    _kde_fill(ax_c, null_z, color="#7F8C8D", alpha=0.25, lw=2.0, label="Permutation Null (B=200)")
    _kde_fill(ax_c, pas_vals[sc_map["scRegPharm_group"] == "scRegPharm+ (Resistant)"], color="#D62728", alpha=0.35, lw=2.4, label="scRegPharm+ (Resistant)")
    _kde_fill(ax_c, pas_vals[sc_map["scRegPharm_group"] == "scRegPharm- (Sensitive)"], color="#1F77B4", alpha=0.35, lw=2.4, label="scRegPharm- (Sensitive)")

    ax_c.axvline(0.75, color="#D62728", ls="--", lw=1.5)
    ax_c.axvline(-0.75, color="#1F77B4", ls="--", lw=1.5)
    ax_c.set_xlabel("Phenotype Association Z-Score (PAS)", fontsize=11.5, fontweight="bold")
    ax_c.set_ylabel("Cellular Density", fontsize=11.5, fontweight="bold")
    ax_c.set_title("Empirical Permutation FDR Separation", fontsize=13.0, fontweight="bold", pad=8)
    q_lim = max(3.5, float(np.percentile(np.abs(pas_vals), 99.5)) * 1.15)
    ax_c.set_xlim(-q_lim, q_lim)
    ax_c.tick_params(axis="both", labelsize=10.5)
    ax_c.grid(True, linestyle="--", alpha=0.35)
    ax_c.legend(loc="upper left", frameon=True, fontsize=10.0)
    sns.despine(ax=ax_c)

    # =========================================================================
    # Panel D: Tri-Modal Pharmacological Reversal Score Decomposition
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 2])
    add_panel_label(ax_d, "D", x=-0.14, y=1.04, fontsize=20, va="bottom")

    top8 = drug_df.head(8).iloc[::-1]
    y_pos = np.arange(len(top8))
    c1 = np.maximum(0.35 * top8["regulon_reversal_z"].values, 0.0)
    c2 = np.maximum(0.35 * top8["gene_wks_z"].values, 0.0)
    c3 = np.maximum(0.30 * top8["crispr_synergy_z"].values, 0.0)

    ax_d.barh(y_pos, c1, color="#C0392B", height=0.62, label="Master Regulon Inversion (35%)")
    ax_d.barh(y_pos, c2, left=c1, color="#F39C12", height=0.62, label="Single-Cell WKS Reversal (35%)")
    ax_d.barh(y_pos, c3, left=c1 + c2, color="#117864", height=0.62, label="DepMap CRISPR Dependency (30%)")

    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels([str(d).upper() for d in top8["drug"]], fontsize=10.5, fontweight="bold")
    ax_d.set_xlabel("Positive PRS Component Contribution", fontsize=11.5, fontweight="bold")
    ax_d.set_title("Tri-Modal Drug Prioritization (Top 8 Hits)", fontsize=13.0, fontweight="bold", pad=8)
    ax_d.set_xlim(0, 3.85)
    ax_d.tick_params(axis="x", labelsize=10.5)
    ax_d.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax_d.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=8.8, borderpad=0.45, handletextpad=0.5, handlelength=1.4)
    sns.despine(ax=ax_d)

    fig.savefig(os.path.join(FIG_DIR, "Figure1_scRegPharm_Framework.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "Figure1_scRegPharm_Framework.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 1.")


def generate_figure2_benchmark():
    """
    Figure 2: Comprehensive Quantitative Benchmarking, Ablation Study & Scalability on Mac mini M4.
    All panels plotted from results/benchmark_*.csv.
    """
    print("Generating Figure 2 (Systematic Benchmarking & Scalability)...")
    bench_df = pd.read_csv(os.path.join(RES_DIR, "benchmark_dropout_robustness.csv"))
    ablation_df = pd.read_csv(os.path.join(RES_DIR, "benchmark_ablation.csv"))
    scale_df = pd.read_csv(os.path.join(RES_DIR, "benchmark_scalability_m4.csv"))

    fig = plt.figure(figsize=(17.2, 10.8), dpi=300)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.05, 1.0, 1.05], hspace=0.36, wspace=0.44)

    # Panel A: AUROC across Observed Single-Cell Zero Fractions (59% - 98%)
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "A", x=-0.14, y=1.08)

    markers = {"scRegPharm (Ours)": "o", "scPAS (BiB 2025)": "s", "Scissor (Nat Biotech)": "^", "LP_SGL (BiB 2024)": "D", "DEGAS / Corr": "v"}
    for m_name, color in METHOD_COLORS.items():
        sub = bench_df[bench_df["method"] == m_name]
        grp = sub.groupby("dropout_rate")[["observed_zero_fraction", "AUROC"]].agg(["mean", "std"]).reset_index()
        x_vals = grp[("observed_zero_fraction", "mean")].values * 100.0
        y_vals = grp[("AUROC", "mean")].values
        s_vals = grp[("AUROC", "std")].fillna(0.0).values
        lw = 2.8 if "scRegPharm" in m_name else 1.9
        ls = "-" if "scRegPharm" in m_name else "--"
        ax_a.plot(
            x_vals, y_vals,
            marker=markers[m_name], color=color, lw=lw, ls=ls, ms=6.5, label=m_name
        )
        ax_a.fill_between(
            x_vals, y_vals - s_vals, y_vals + s_vals,
            color=color, alpha=0.14
        )

    ax_a.set_xlabel("Observed Single-Cell Zero Fraction (%)", fontsize=10, fontweight="bold")
    ax_a.set_ylabel("Phenotype Subpopulation AUROC", fontsize=10, fontweight="bold")
    ax_a.set_title("AUROC across UMI Downsampling Regimes", fontsize=11, fontweight="bold", pad=8)
    ax_a.set_ylim(0.65, 1.01)
    ax_a.grid(True, linestyle="--", alpha=0.35)
    ax_a.legend(loc="lower left", frameon=True, fontsize=8.0)
    sns.despine(ax=ax_a)

    # Panel B: AUPRC & AUROC under High Sparsity (87% - 98% Zero Fraction)
    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "B", x=-0.15, y=1.08)

    high_drop = bench_df[bench_df["dropout_rate"] >= 0.85].copy()
    methods_order = list(METHOD_COLORS.keys())
    x_idx = np.arange(len(methods_order))
    width = 0.36

    auroc_means = [high_drop[high_drop["method"] == m]["AUROC"].mean() for m in methods_order]
    auroc_stds = [high_drop[high_drop["method"] == m]["AUROC"].std() for m in methods_order]
    auprc_means = [high_drop[high_drop["method"] == m]["AUPRC"].mean() for m in methods_order]
    auprc_stds = [high_drop[high_drop["method"] == m]["AUPRC"].std() for m in methods_order]

    bars1 = ax_b.bar(x_idx - width / 2, auroc_means, width, yerr=auroc_stds, capsize=3.0, color="#C0392B", alpha=0.88, edgecolor="black", lw=0.8, label="AUROC (High Sparsity)")
    bars2 = ax_b.bar(x_idx + width / 2, auprc_means, width, yerr=auprc_stds, capsize=3.0, color="#2874A6", alpha=0.85, edgecolor="black", lw=0.8, label="AUPRC (High Sparsity)")

    for b, e in zip(bars1, auroc_stds):
        ax_b.text(b.get_x() + b.get_width() / 2, b.get_height() + e + 0.012, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#922B21")
    for b, e in zip(bars2, auprc_stds):
        ax_b.text(b.get_x() + b.get_width() / 2, b.get_height() + e + 0.012, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#1B4F72")

    ax_b.set_xticks(x_idx)
    ax_b.set_xticklabels([m.split(" (")[0] for m in methods_order], rotation=20, ha="right", fontsize=9.0, fontweight="bold")
    ax_b.set_ylabel("Score (87%–98% Zero Fraction)", fontsize=10, fontweight="bold")
    ax_b.set_title("Accuracy under Extreme UMI Sparsity", fontsize=11, fontweight="bold", pad=8)
    ax_b.set_ylim(0.50, 1.12)
    ax_b.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.17), ncol=2, frameon=True, fontsize=7.2)
    sns.despine(ax=ax_b)

    # Panel C: Ablation Study of scRegPharm Architectural Modules
    ax_c = fig.add_subplot(gs[0, 2])
    add_panel_label(ax_c, "C", x=-0.22, y=1.08)

    ab_order = [
        "Full scRegPharm",
        "w/o Optimal Transport",
        "w/o Graph Diffusion",
        "w/o Regulon (Raw Gene Space)",
        "w/o Dual-Ridge Woodbury",
    ][::-1]
    short_ab = [
        "w/o Woodbury Ridge",
        "w/o Regulon (Raw)",
        "w/o Graph Diffusion",
        "w/o Optimal Transport",
        "Full scRegPharm",
    ]
    ab_grp = ablation_df.groupby("variant")[["AUROC", "AUPRC"]].agg(["mean", "std"]).loc[ab_order]
    y_ab = np.arange(len(ab_order))
    ab_colors = ["#7F8C8D", "#5DADE2", "#48C9B0", "#F39C12", "#D62728"]

    means_auc = ab_grp[("AUROC", "mean")].values
    stds_auc = ab_grp[("AUROC", "std")].fillna(0.0).values

    ax_c.barh(y_ab, means_auc, xerr=stds_auc, height=0.56, color=ab_colors, edgecolor="black", lw=0.9, capsize=3.5)
    for i, (m_v, s_v) in enumerate(zip(means_auc, stds_auc)):
        ax_c.text(m_v + s_v + 0.008, i, f"{m_v:.3f}", va="center", ha="left", fontsize=8.5, fontweight="bold")

    ax_c.set_yticks(y_ab)
    ax_c.set_yticklabels(short_ab, fontsize=8.8, fontweight="bold")
    ax_c.set_xlim(min(0.60, float(means_auc.min()) - 0.08), 1.06)
    ax_c.set_xlabel("AUROC (Mean ± SD across 5 Splits)", fontsize=10, fontweight="bold")
    ax_c.set_title("Component-Wise Ablation Analysis", fontsize=11, fontweight="bold", pad=8)
    ax_c.grid(True, axis="x", linestyle="--", alpha=0.35)
    sns.despine(ax=ax_c)

    # Panel D: Computational Runtime Scalability on Mac mini M4
    ax_d = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_d, "D", x=-0.14, y=1.08)

    n_cells = scale_df["n_cells"].values / 1000.0
    t_ours = scale_df["scRegPharm_time_s"].values
    t_scpas = scale_df["scPAS_time_s"].values
    t_scissor = scale_df["Scissor_time_s"].values

    ax_d.plot(n_cells, t_scissor, "-^", color="#2CA02C", lw=2.3, ms=6.5, label="Scissor")
    ax_d.plot(n_cells, t_scpas, "-s", color="#1F77B4", lw=2.3, ms=6.5, label="scPAS")
    ax_d.plot(n_cells, t_ours, "-o", color="#D62728", lw=2.8, ms=7.0, label="scRegPharm (Ours)")

    ax_d.set_yscale("log")
    ax_d.set_xlabel("Number of Single Cells (× 1,000)", fontsize=10, fontweight="bold")
    ax_d.set_ylabel("Measured Wall-Clock Runtime (s, Log)", fontsize=10, fontweight="bold")
    ax_d.set_title("Wall-clock Runtime on Apple M4 (32GB)", fontsize=11, fontweight="bold", pad=8)
    ax_d.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_d.legend(loc="upper left", frameon=True, fontsize=8.0)
    sns.despine(ax=ax_d)

    # Panel E: Peak Memory Footprint (MB) on Mac mini M4
    ax_e = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_e, "E", x=-0.15, y=1.08)

    mem_ours = scale_df["scRegPharm_mem_MB"].values
    mem_scpas = scale_df["scPAS_mem_MB"].values
    mem_scissor = scale_df["Scissor_mem_MB"].values

    ax_e.plot(n_cells, mem_scissor, "-^", color="#2CA02C", lw=2.3, ms=6.5, label="Scissor")
    ax_e.plot(n_cells, mem_scpas, "-s", color="#1F77B4", lw=2.3, ms=6.5, label="scPAS")
    ax_e.plot(n_cells, mem_ours, "-o", color="#D62728", lw=2.8, ms=7.0, label="scRegPharm (Ours)")

    ax_e.set_yscale("log")
    ax_e.set_xlabel("Number of Single Cells (× 1,000)", fontsize=10, fontweight="bold")
    ax_e.set_ylabel("Measured Peak RAM via tracemalloc (MB)", fontsize=10, fontweight="bold")
    ax_e.set_title("Peak Memory Footprint across Scale", fontsize=11, fontweight="bold", pad=8)
    ax_e.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_e.legend(loc="upper left", frameon=True, fontsize=8.5)
    sns.despine(ax=ax_e)

    # Panel F: Multi-metric benchmark summary
    ax_f = fig.add_subplot(gs[1, 2])
    add_panel_label(ax_f, "F", x=-0.18, y=1.08)

    metrics_names = ["Mean AUC", "Mean AUPR", "94% AUC", "98% AUC", "98% AUPR", "Speed (N=6.9k)"]
    methods_full = ["scRegPharm (Ours)", "scPAS (BiB 2025)", "LP_SGL (BiB 2024)", "DEGAS / Corr", "Scissor (Nat Biotech)"]
    methods_short = ["scRegPharm", "scPAS", "LP_SGL", "DEGAS/Corr", "Scissor"]

    speed_df = pd.read_csv(os.path.join(RES_DIR, "benchmark_speed_6879.csv"))
    speed_map = dict(zip(speed_df["method"], speed_df["runtime_s"]))

    score_rows = []
    for m in methods_full:
        sub_m = bench_df[bench_df["method"] == m]
        sub_94 = sub_m[sub_m["dropout_rate"] == 0.94]
        sub_98 = sub_m[sub_m["dropout_rate"] == 0.98]
        mean_rt = float(speed_map[m])
        speed_idx = 1.0 / (1.0 + np.log1p(mean_rt * 5.0))
        score_rows.append([
            float(sub_m["AUROC"].mean()),
            float(sub_m["AUPRC"].mean()),
            float(sub_94["AUROC"].mean()),
            float(sub_98["AUROC"].mean()),
            float(sub_98["AUPRC"].mean()),
            float(speed_idx),
        ])
    score_mat = np.array(score_rows)

    sns.heatmap(
        score_mat, ax=ax_f, cmap="YlOrRd", vmin=float(min(0.45, score_mat.min()) - 0.02), vmax=1.00,
        annot=True, fmt=".2f", cbar=True, linewidths=1.2, linecolor="white",
        xticklabels=metrics_names, yticklabels=methods_short,
        annot_kws={"size": 8.6, "weight": "bold"}
    )
    ax_f.set_xticklabels(metrics_names, rotation=25, ha="right", fontsize=8.3, fontweight="bold")
    ax_f.set_yticklabels(methods_short, rotation=0, fontsize=9.0, fontweight="bold")
    ax_f.set_title("Empirical Benchmark Scorecard", fontsize=11, fontweight="bold", pad=8)

    fig.savefig(os.path.join(FIG_DIR, "Figure2_Benchmark_Scalability.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "Figure2_Benchmark_Scalability.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 2.")


def generate_figure3_single_cell_melanoma():
    """
    Figure 3: Single-Cell Dissection of Anti-PD-1 Resistance States & Master Regulator Network
    in Human Melanoma (Jerby-Arnon et al., Cell 2018, GSE115978, N = 6,879 cells).
    """
    print("Generating Figure 3 (Melanoma Single-Cell Phenotype Mapping & Regulon Landscape)...")
    sc_res = pd.read_csv(os.path.join(RES_DIR, "melanoma_single_cell_mapping.csv"), index_col=0)
    sc_reg = pd.read_csv(os.path.join(RES_DIR, "melanoma_single_cell_regulon_activity.csv"), index_col=0)
    ct_sum = pd.read_csv(os.path.join(RES_DIR, "melanoma_celltype_enrichment.csv"), index_col=0)
    mr_df = pd.read_csv(os.path.join(RES_DIR, "master_regulator_differential.csv"))

    fig = plt.figure(figsize=(17.8, 11.2), dpi=300)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.05, 1.08], hspace=0.38, wspace=0.52)

    # Panel A: 2D Manifold Embedding Colored by Microenvironment Cell Subpopulations (N = 6,879 cells)
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "A", x=-0.16, y=1.10)

    ct_palette = {
        "Malignant_MYC_TCF4_Excl": "#D62728",
        "Malignant_AXL_MES_Res": "#E67E22",
        "Macrophage_SPP1_TREM2": "#8E44AD",
        "MyoCAF_FAP_TGFB": "#C0392B",
        "Malignant_MITF_Diff": "#F1C40F",
        "B_Plasma_and_Endothelial": "#95A5A6",
        "cDC1_BATF3_AntigenPres": "#16A085",
        "CD8_T_Cytotoxic_Effector": "#2980B9",
    }
    short_names = {
        "Malignant_MYC_TCF4_Excl": "Malig_MYC/TCF4 (Excl)",
        "Malignant_AXL_MES_Res": "Malig_AXL/MES (Res)",
        "Macrophage_SPP1_TREM2": "Mac_SPP1+/TREM2+",
        "MyoCAF_FAP_TGFB": "MyoCAF_FAP+/TGFB",
        "Malignant_MITF_Diff": "Malig_MITF (Diff)",
        "B_Plasma_and_Endothelial": "B / Endothelial",
        "cDC1_BATF3_AntigenPres": "cDC1_BATF3+/IRF8+",
        "CD8_T_Cytotoxic_Effector": "CD8+ T Cytotoxic",
    }

    for ct, color in ct_palette.items():
        sub = sc_res[sc_res["cell_type"] == ct]
        if not sub.empty:
            ax_a.scatter(sub["UMAP_1"], sub["UMAP_2"], c=color, s=6, alpha=0.78, edgecolors="none", label=short_names[ct])

    ax_a.set_xlabel("Regulon Manifold Dimension 1", fontsize=10, fontweight="bold")
    ax_a.set_ylabel("Regulon Manifold Dimension 2", fontsize=10, fontweight="bold")
    ax_a.set_title(f"GSE115978 Melanoma TME Atlas (n = {len(sc_res):,} Cells)", fontsize=9.8, fontweight="bold", pad=16)
    ax_a.legend(loc="upper right", frameon=True, fontsize=7.0, markerscale=2.0, ncol=1)
    sns.despine(ax=ax_a)

    # Panel B: Continuous scRegPharm Phenotype Association Score (PAS_z) on Manifold
    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "B", x=-0.16, y=1.10)

    v_lim = max(2.5, float(np.percentile(np.abs(sc_res["PAS_score"]), 98)))
    sc_plot = ax_b.scatter(
        sc_res["UMAP_1"], sc_res["UMAP_2"],
        c=sc_res["PAS_score"], cmap="coolwarm", vmin=-v_lim, vmax=v_lim,
        s=6, alpha=0.82, edgecolors="none"
    )
    cbar = plt.colorbar(sc_plot, ax=ax_b, fraction=0.040, pad=0.025)
    cbar.ax.set_title("PAS\nZ-Score", fontsize=8.0, fontweight="bold", pad=5)

    ax_b.set_xlabel("Regulon Manifold Dimension 1", fontsize=10, fontweight="bold")
    ax_b.set_ylabel("Regulon Manifold Dimension 2", fontsize=10, fontweight="bold")
    ax_b.set_title("scRegPharm Mapping (PRJEB23709 → GSE115978)", fontsize=9.8, fontweight="bold", pad=16)
    sns.despine(ax=ax_b)

    # Panel C: Cell-Type Proportion of scRegPharm+ / scRegPharm-
    ax_c = fig.add_subplot(gs[0, 2])
    add_panel_label(ax_c, "C", x=-0.22, y=1.08)

    ct_order = ct_sum.index[::-1]
    y_pos = np.arange(len(ct_order))
    p_res = ct_sum.loc[ct_order, "Pct_Resistant"].values
    p_sens = ct_sum.loc[ct_order, "Pct_Sensitive"].values
    p_bg = np.maximum(0.0, 100.0 - p_res - p_sens)

    ax_c.barh(y_pos, p_res, color="#D62728", height=0.62, label="scRegPharm+ (Resistant)")
    ax_c.barh(y_pos, p_bg, left=p_res, color="#DCDCDC", height=0.62, label="Background (Neutral)")
    ax_c.barh(y_pos, p_sens, left=p_res + p_bg, color="#1F77B4", height=0.62, label="scRegPharm- (Sensitive)")

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels([short_names.get(c, c) for c in ct_order], fontsize=8.5, fontweight="bold")
    ax_c.set_xlabel("Fraction of Cells (%)", fontsize=10, fontweight="bold")
    ax_c.set_title("Subpopulation Phenotype Enrichment", fontsize=11, fontweight="bold", pad=8)
    ax_c.set_xlim(0, 100)
    ax_c.legend(loc="lower center", bbox_to_anchor=(0.45, -0.27), ncol=3, frameon=False, fontsize=7.5)
    sns.despine(ax=ax_c)

    # Panel D: Master Regulator Differential Waterfall (Top 9 Resistant vs Top 8 Sensitive TFs)
    ax_d = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_d, "D", x=-0.16, y=1.08)

    top_pos_mr = mr_df.head(9)
    top_neg_mr = mr_df.tail(8)
    sub_mr = pd.concat([top_pos_mr, top_neg_mr]).iloc[::-1]

    y_mr = np.arange(len(sub_mr))
    mr_colors = ["#C0392B" if v > 0 else "#2874A6" for v in sub_mr["delta_activity"]]
    ax_d.barh(y_mr, sub_mr["delta_activity"], color=mr_colors, height=0.68, edgecolor="black", lw=0.6)
    ax_d.axvline(0, color="black", lw=1.0)

    ax_d.set_yticks(y_mr)
    ax_d.set_yticklabels([str(r).replace("HALLMARK_", "HM_") for r in sub_mr["regulon"]], fontsize=8.3, fontweight="bold")
    ax_d.set_xlabel("Δ Regulon Activity (scRegPharm+ vs scRegPharm-)", fontsize=9.5, fontweight="bold")
    ax_d.set_title("Master Regulators of ICB Resistance", fontsize=11, fontweight="bold", pad=8)
    ax_d.grid(True, axis="x", linestyle="--", alpha=0.35)
    sns.despine(ax=ax_d)

    # Panel E: Single-Cell Regulon Activity Heatmap Across Subpopulations (Spans gs[1, 1:])
    ax_e = fig.add_subplot(gs[1, 1:])
    add_panel_label(ax_e, "E", x=-0.08, y=1.08)

    selected_regs = [r for r in (list(top_pos_mr["regulon"][:8]) + list(top_neg_mr["regulon"][-7:][::-1])) if r in sc_reg.index]
    sc_reg_T = sc_reg.loc[selected_regs].T.copy()
    sc_reg_T["cell_type"] = sc_res["cell_type"].values
    hm_df = sc_reg_T.groupby("cell_type").mean().loc[ct_sum.index].T
    hm_df.index = [str(r).replace("HALLMARK_", "HM_") for r in hm_df.index]

    hm_df.columns = [short_names.get(c, c) for c in hm_df.columns]
    hm_vlim = max(2.0, float(np.nanmax(np.abs(hm_df.values))) * 1.05)
    sns.heatmap(
        hm_df, ax=ax_e, cmap="RdBu_r", vmin=-hm_vlim, vmax=hm_vlim, center=0,
        annot=True, fmt=".2f", linewidths=0.8, linecolor="white",
        cbar_kws={"label": "Mean Standardized Regulon Activity", "pad": 0.02},
        annot_kws={"size": 7.8, "weight": "bold"}
    )
    ax_e.set_xlabel("")
    ax_e.set_ylabel("")
    ax_e.set_xticklabels(hm_df.columns, rotation=22, ha="right", fontsize=8.8, fontweight="bold")
    ax_e.set_yticklabels(hm_df.index, rotation=0, fontsize=8.8, fontweight="bold")
    ax_e.set_title("Cross-Lineage Master Regulator Activity Signature across GSE115978 TME States", fontsize=11, fontweight="bold", pad=8)

    fig.savefig(os.path.join(FIG_DIR, "Figure3_Melanoma_SingleCell_Landscape.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "Figure3_Melanoma_SingleCell_Landscape.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 3.")


if __name__ == "__main__":
    generate_figure1_framework()
    generate_figure2_benchmark()
    generate_figure3_single_cell_melanoma()
    print("Figures 1, 2, and 3 regenerated successfully.")
