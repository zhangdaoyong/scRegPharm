"""
Generate Figures 4, 5, and 6 (300 DPI PNG and vector PDF) from results/*.csv.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
from lifelines import KaplanMeierFitter

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


def add_panel_label(ax, label, x=-0.12, y=1.08, fontsize=15):
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold", va="top", ha="left", color="#1A252F"
    )


def generate_figure4_spatial():
    """
    Figure 4: 10x Visium Spatial Transcriptomics Architecture (GSE298774, GSM9022963_A17, S = 1,461 spots)
    & Fibro-Myeloid Immune Exclusion Barrier.
    """
    print("Generating Figure 4 (10x Visium Spatial Transcriptomics & Immune Exclusion)...")
    sp_res = pd.read_csv(os.path.join(RES_DIR, "visium_spatial_results.csv"), index_col=0)

    fig = plt.figure(figsize=(17.2, 11.2), dpi=300)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.05, 1.0], width_ratios=[1.05, 1.0, 1.0], hspace=0.38, wspace=0.42)

    # Panel A: Spatial Molecular Niche Domains on the 10x Visium Array (S = 1,461 in-tissue spots)
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "A", x=-0.16, y=1.10)

    dom_colors = {
        "Tumor_Core_MYC_SOX10": "#C0392B",
        "FibroMyeloid_Boundary_SPP1_CAF": "#E67E22",
        "Excluded_Immune_CD8_MHCII": "#2980B9",
        "Peritumoral_Stroma": "#BDC3C7",
    }
    dom_short = {
        "Tumor_Core_MYC_SOX10": "Tumor Core (MYC/SOX10+)",
        "FibroMyeloid_Boundary_SPP1_CAF": "Fibro-Myeloid Boundary",
        "Excluded_Immune_CD8_MHCII": "Excluded Immune (CD8/MHCII)",
        "Peritumoral_Stroma": "Peritumoral Stroma",
    }

    for dom, col in dom_colors.items():
        sub = sp_res[sp_res["spatial_domain"] == dom]
        if not sub.empty:
            ax_a.scatter(sub["x"], sub["y"], c=col, s=16, marker="h", alpha=0.90, edgecolors="none", label=dom_short[dom])

    ax_a.set_aspect("equal")
    ax_a.set_xlabel("Visium Array Coordinate X", fontsize=9.5, fontweight="bold")
    ax_a.set_ylabel("Visium Array Coordinate Y", fontsize=9.5, fontweight="bold")
    ax_a.set_title(f"GSE298774 Spatial Niche Architecture (n = {len(sp_res):,} Spots)", fontsize=9.8, fontweight="bold", pad=16)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, fontsize=7.5, markerscale=1.3)
    sns.despine(ax=ax_a)

    # Panel B: Spatially Smoothed scRegPharm Resistance Score (sPAS)
    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "B", x=-0.14, y=1.08)

    v_spas = max(1.8, float(np.percentile(np.abs(sp_res["sPAS_score"]), 97)))
    p_b = ax_b.scatter(
        sp_res["x"], sp_res["y"], c=sp_res["sPAS_score"],
        cmap="RdBu_r", vmin=-v_spas, vmax=v_spas, s=16, marker="h", edgecolors="none"
    )
    cb_b = plt.colorbar(p_b, ax=ax_b, fraction=0.042, pad=0.03)
    cb_b.set_label("sPAS Z-Score", fontsize=8.5, fontweight="bold")
    ax_b.set_aspect("equal")
    ax_b.set_xlabel("Visium Array Coordinate X", fontsize=9.5, fontweight="bold")
    ax_b.set_ylabel("Visium Array Coordinate Y", fontsize=9.5, fontweight="bold")
    ax_b.set_title("scRegPharm-Spatial Resistance Score", fontsize=11, fontweight="bold", pad=8)
    sns.despine(ax=ax_b)

    # Panel C: Fibro-Myeloid Barrier Module Score (SPP1+ TAM / TGFB+ MyoCAF / SMAD3)
    ax_c = fig.add_subplot(gs[0, 2])
    add_panel_label(ax_c, "C", x=-0.14, y=1.08)

    p_c = ax_c.scatter(
        sp_res["x"], sp_res["y"], c=sp_res["Barrier_SPP1_CAF_score"],
        cmap="YlOrBr", vmin=float(np.percentile(sp_res["Barrier_SPP1_CAF_score"], 5)),
        vmax=float(np.percentile(sp_res["Barrier_SPP1_CAF_score"], 95)),
        s=16, marker="h", edgecolors="none"
    )
    cb_c = plt.colorbar(p_c, ax=ax_c, fraction=0.042, pad=0.03)
    cb_c.set_label("Barrier Module Activity", fontsize=8.5, fontweight="bold")
    ax_c.set_aspect("equal")
    ax_c.set_xlabel("Visium Array Coordinate X", fontsize=9.5, fontweight="bold")
    ax_c.set_ylabel("Visium Array Coordinate Y", fontsize=9.5, fontweight="bold")
    ax_c.set_title("Fibro-Myeloid Boundary Activity", fontsize=11, fontweight="bold", pad=8)
    sns.despine(ax=ax_c)

    # Panel D: Cytotoxic CD8+ T / Antigen Presentation Effector Score
    ax_d = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_d, "D", x=-0.14, y=1.08)

    p_d = ax_d.scatter(
        sp_res["x"], sp_res["y"], c=sp_res["Effector_CD8_cDC1_score"],
        cmap="GnBu", vmin=float(np.percentile(sp_res["Effector_CD8_cDC1_score"], 5)),
        vmax=float(np.percentile(sp_res["Effector_CD8_cDC1_score"], 95)),
        s=16, marker="h", edgecolors="none"
    )
    cb_d = plt.colorbar(p_d, ax=ax_d, fraction=0.042, pad=0.03)
    cb_d.set_label("Effector Module Score", fontsize=8.5, fontweight="bold")
    ax_d.set_aspect("equal")
    ax_d.set_xlabel("Visium Array Coordinate X", fontsize=9.5, fontweight="bold")
    ax_d.set_ylabel("Visium Array Coordinate Y", fontsize=9.5, fontweight="bold")
    ax_d.set_title("Excluded Peritumoral Effector Immunity", fontsize=11, fontweight="bold", pad=8)
    sns.despine(ax=ax_d)

    # Panel E: Radial Spatial Gradient Profile across Visium Coordinates
    ax_e = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_e, "E", x=-0.14, y=1.08)

    r_min = float(np.percentile(sp_res["radial_distance"], 2))
    r_max = float(np.percentile(sp_res["radial_distance"], 96))
    sp_res["r_bin"] = pd.cut(sp_res["radial_distance"], bins=np.linspace(r_min, r_max, 16), labels=False)
    rad_grp = sp_res.groupby("r_bin")[["radial_distance", "Malignant_MYC_AXL_score", "Barrier_SPP1_CAF_score", "Effector_CD8_cDC1_score"]].agg(["mean", "sem"]).dropna()

    r_x = rad_grp[("radial_distance", "mean")].values
    for col_name, label_str, color in [
        ("Malignant_MYC_AXL_score", "Malignant Core (MYC/SOX10)", "#C0392B"),
        ("Barrier_SPP1_CAF_score", "SPP1+ TAM / MyoCAF Boundary", "#E67E22"),
        ("Effector_CD8_cDC1_score", "CD8+ T / MHC-II Effector", "#2980B9"),
    ]:
        m_v = rad_grp[(col_name, "mean")].values
        s_v = rad_grp[(col_name, "sem")].values * 1.96
        ax_e.plot(r_x, m_v, lw=2.4, color=color, label=label_str)
        ax_e.fill_between(r_x, m_v - s_v, m_v + s_v, color=color, alpha=0.18)

    ax_e.axhline(0, color="#7F8C8D", ls="--", lw=1.0)
    ax_e.set_xlabel("Radial Distance from Tumor Core Centroid (Array Units)", fontsize=9.2, fontweight="bold")
    ax_e.set_ylabel("Standardized Regulon Module Activity", fontsize=9.5, fontweight="bold")
    ax_e.set_title("Radial Spatial Exclusion Gradient", fontsize=11, fontweight="bold", pad=8)
    ax_e.grid(True, linestyle="--", alpha=0.35)
    ax_e.legend(loc="best", frameon=True, fontsize=7.6)
    sns.despine(ax=ax_e)

    # Panel F: Spatial Domain Distribution & Empirical Exclusion Statistics
    ax_f = fig.add_subplot(gs[1, 2])
    add_panel_label(ax_f, "F", x=-0.14, y=1.08)

    dom_order = [
        "Tumor_Core_MYC_SOX10",
        "FibroMyeloid_Boundary_SPP1_CAF",
        "Peritumoral_Stroma",
        "Excluded_Immune_CD8_MHCII",
    ]
    dom_order = [d for d in dom_order if d in sp_res["spatial_domain"].unique()]
    short_labels_map = {
        "Tumor_Core_MYC_SOX10": "Tumor Core",
        "FibroMyeloid_Boundary_SPP1_CAF": "Fibro-Myeloid\nBoundary",
        "Peritumoral_Stroma": "Peritumoral\nStroma",
        "Excluded_Immune_CD8_MHCII": "Excluded\nImmune",
    }
    short_labels = [short_labels_map[d] for d in dom_order]
    pal_list = [dom_colors[d] for d in dom_order]

    sns.violinplot(
        data=sp_res, x="spatial_domain", y="sPAS_score", hue="spatial_domain",
        order=dom_order, hue_order=dom_order,
        palette=pal_list, inner="box", cut=0, ax=ax_f, linewidth=1.1, legend=False
    )
    r_val, p_pear = stats.pearsonr(sp_res["Malignant_MYC_AXL_score"], sp_res["Effector_CD8_cDC1_score"])
    groups_kw = [sp_res.loc[sp_res["spatial_domain"] == d, "sPAS_score"].values for d in dom_order]
    _, p_kw = stats.kruskal(*groups_kw)
    p_pear_str = f"{p_pear:.1e}" if p_pear > 1e-300 else "< 1e-300"
    p_kw_str = f"{p_kw:.1e}" if p_kw > 1e-300 else "< 1e-300"

    ax_f.text(
        0.04, 0.06,
        f"Tumor Core vs. CD8+ Effector:\nPearson r = {r_val:.3f} (P = {p_pear_str})\nKruskal-Wallis P = {p_kw_str}",
        transform=ax_f.transAxes, fontsize=8.0, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9FA", edgecolor="#BDC3C7")
    )
    ax_f.axhline(0, color="#34495E", ls="--", lw=1.0)
    ax_f.set_xticks(range(len(short_labels)))
    ax_f.set_xticklabels(short_labels, fontsize=8.5, fontweight="bold")
    ax_f.set_xlabel("")
    ax_f.set_ylabel("scRegPharm Spatial Score (sPAS)", fontsize=9.5, fontweight="bold")
    ax_f.set_title("Domain Stratification & T-Cell Exclusion", fontsize=11, fontweight="bold", pad=8)
    ax_f.grid(True, axis="y", linestyle="--", alpha=0.35)
    sns.despine(ax=ax_f)

    fig.savefig(os.path.join(FIG_DIR, "Figure4_Visium_Spatial_Exclusion.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "Figure4_Visium_Spatial_Exclusion.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 4.")


def generate_figure5_drug_repurposing():
    """
    Figure 5: A375 LINCS L1000 Chemical Perturbations + Broad DepMap 2023 Melanoma CRISPR Benchmark.
    """
    print("Generating Figure 5 (A375 LINCS L1000 & DepMap CRISPR Drug Benchmark)...")
    drug_df = pd.read_csv(os.path.join(RES_DIR, "drug_repurposing_rankings.csv"))
    drug_reg = pd.read_csv(os.path.join(RES_DIR, "drug_regulon_perturbation_matrix.csv"), index_col=0)

    fig = plt.figure(figsize=(17.0, 11.6), dpi=300)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05], width_ratios=[1.02, 1.12], hspace=0.55, wspace=0.28)

    gs_top_left = gs[0, 0].subgridspec(1, 2, wspace=0.30)
    ax_a1 = fig.add_subplot(gs_top_left[0, 0])
    ax_a2 = fig.add_subplot(gs_top_left[0, 1])
    add_panel_label(ax_a1, "A", x=-0.22, y=1.10)
    add_panel_label(ax_a2, "B", x=-0.36, y=1.15)

    y_true = drug_df["is_clinical_gold_standard"].values
    # Method scores from drug_repurposing_rankings.csv
    s_ours = drug_df["PRS_score"].values
    s_tx = drug_df["PRS_transcriptome_only"].values
    s_drugresc = drug_df["DrugReSC_score"].values
    s_asgard = drug_df["ASGARD_score"].values
    s_bulkcmap = drug_df["BulkCMap_score"].values

    drug_methods = [
        ("scRegPharm-Drug (tri-modal)", s_ours, "#D62728", 2.8, "-"),
        ("scRegPharm-Drug (transcriptome-only)", s_tx, "#FF7F0E", 2.4, "-"),
        ("DrugReSC (BiB 2024)", s_drugresc, "#1F77B4", 2.1, "--"),
        ("ASGARD", s_asgard, "#2CA02C", 2.1, "-."),
        ("Bulk-CMap", s_bulkcmap, "#7F7F7F", 1.9, ":"),
    ]
    short_lab = {
        "scRegPharm-Drug (tri-modal)": "scRP tri-modal",
        "scRegPharm-Drug (transcriptome-only)": "scRP tx-only",
        "DrugReSC (BiB 2024)": "DrugReSC",
        "ASGARD": "ASGARD",
        "Bulk-CMap": "Bulk-CMap",
    }

    for m_label, scores, color, lw, ls in drug_methods:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_val = roc_auc_score(y_true, scores)
        ax_a1.plot(fpr, tpr, color=color, lw=lw, ls=ls, label=f"{short_lab[m_label]} ({auc_val:.3f})")

    ax_a1.plot([0, 1], [0, 1], color="#BDC3C7", lw=1.2, ls="--")
    ax_a1.set_xlabel("False Positive Rate", fontsize=9.5, fontweight="bold")
    ax_a1.set_ylabel("True Positive Rate", fontsize=9.5, fontweight="bold")
    ax_a1.set_title("A375 LINCS L1000 Discovery ROC", fontsize=10.5, fontweight="bold", pad=8)
    ax_a1.grid(True, linestyle="--", alpha=0.35)
    ax_a1.legend(loc="upper center", bbox_to_anchor=(0.42, -0.15), ncol=3, frameon=False, fontsize=6.2, handlelength=1.0, columnspacing=0.7, handletextpad=0.35)
    sns.despine(ax=ax_a1)

    # Panel B: Top-K Validated Oncology Compounds Recovered (Top 5, Top 10, Top 15, Top 20)
    k_values = [5, 10, 15, 20]
    k_labels = ["Top 5", "Top 10", "Top 15", "Top 20"]
    x_k = np.arange(len(k_labels))
    w_bar = 0.16

    for idx_m, (m_label, scores, color, _, _) in enumerate(drug_methods):
        hits = [int(y_true[np.argsort(scores)[::-1][:k]].sum()) for k in k_values]
        offset = (idx_m - 2.0) * w_bar
        b_rects = ax_a2.bar(
            x_k + offset, hits, w_bar, color=color,
            edgecolor="black", lw=0.7, label=m_label.replace("scRegPharm-Drug ", "scRP ")
        )
        for j, r in enumerate(b_rects):
            ax_a2.text(
                r.get_x() + r.get_width() / 2, r.get_height() + 0.22 + (idx_m % 2) * 1.05,
                f"{int(r.get_height())}", ha="center", va="bottom", fontsize=7.0, fontweight="bold"
            )

    ax_a2.set_xticks(x_k)
    ax_a2.set_xticklabels(k_labels, fontsize=9, fontweight="bold")
    ax_a2.set_ylabel("Benchmark Positives Recovered", fontsize=9.5, fontweight="bold")
    ax_a2.set_title("Top-K Benchmark-Positive Enrichment", fontsize=10.5, fontweight="bold", pad=8)
    ax_a2.set_ylim(0, 21.5)
    ax_a2.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_a2.legend(loc="upper center", bbox_to_anchor=(0.58, -0.14), ncol=3, frameon=False, fontsize=6.2, handlelength=1.0, columnspacing=0.7, handletextpad=0.35)
    sns.despine(ax=ax_a2)

    # Panel C: 2D Pharmacological Reversal (PRS) vs Broad DepMap 2023 Melanoma CRISPR Essentiality
    ax_c = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_c, "C", x=-0.15, y=1.12)

    neg_drugs = drug_df[drug_df["is_clinical_gold_standard"] == 0]
    pos_drugs = drug_df[drug_df["is_clinical_gold_standard"] == 1]

    ax_c.scatter(
        neg_drugs["PRS_score"], -neg_drugs["target_CRISPR_chronos"],
        s=75, c="#95A5A6", alpha=0.68, edgecolors="#2C3E50", lw=0.8,
        label=f"Non-Oncology / Control Compounds (n = {len(neg_drugs)})"
    )
    ax_c.scatter(
        pos_drugs["PRS_score"], -pos_drugs["target_CRISPR_chronos"],
        s=115, c="#D62728", alpha=0.90, edgecolors="black", lw=1.1,
        label=f"Mechanism-Annotated Anticancer Positives (n = {len(pos_drugs)})"
    )

    ax_c.axvline(0.50, color="#C0392B", ls="--", lw=1.2)
    ax_c.axhline(0.40, color="#117864", ls="--", lw=1.2)

    # Annotate top-ranked compounds
    top10 = drug_df.head(10).copy()
    top10["y_dep"] = -top10["target_CRISPR_chronos"].astype(float)
    x_min = float(drug_df["PRS_score"].min())
    x_max = float(drug_df["PRS_score"].max())
    y_min = float((-drug_df["target_CRISPR_chronos"]).min())
    y_max = float((-drug_df["target_CRISPR_chronos"]).max())
    x_lo, x_hi = x_min - 0.60, x_max + 0.65
    y_band1 = y_min - 0.24
    y_band2 = y_min - 0.47
    half = (len(top10) + 1) // 2
    for k, (_, row) in enumerate(top10.sort_values("PRS_score", ascending=False).iterrows()):
        d_name = str(row["drug"])
        x_v = float(row["PRS_score"])
        y_v = float(row["y_dep"])
        row_i, within = (0, k) if k < half else (1, k - half)
        x_lab = x_lo + (within + 0.5) / half * (x_hi - x_lo)
        y_lab = y_band1 if row_i == 0 else y_band2
        ax_c.annotate(
            d_name.upper(), (x_v, y_v), xytext=(x_lab, y_lab), textcoords="data",
            fontsize=7.0, fontweight="bold", color="#1A252F",
            ha="center", va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#D5D8DC", lw=0.5, alpha=0.95, zorder=4),
            arrowprops=dict(arrowstyle="-", color="#95A5A6", lw=0.6, shrinkA=8, shrinkB=2)
        )

    x_min, x_max = float(drug_df["PRS_score"].min()), float(drug_df["PRS_score"].max())
    y_min, y_max = float((-drug_df["target_CRISPR_chronos"]).min()), float((-drug_df["target_CRISPR_chronos"]).max())
    ax_c.set_xlim(x_min - 0.60, x_max + 0.65)
    ax_c.set_ylim(y_min - 0.62, y_max + 0.30)
    ax_c.set_xlabel("scRegPharm Pharmacological Reversal Score (PRS)", fontsize=10, fontweight="bold")
    ax_c.set_ylabel("Target Dependency (DepMap melanoma)", fontsize=10, fontweight="bold")
    ax_c.set_title("A375 L1000 Reversal vs. DepMap Target Dependency (direct targets only)", fontsize=10.8, fontweight="bold", pad=8)
    ax_c.grid(True, linestyle="--", alpha=0.35)
    ax_c.legend(loc="upper left", frameon=True, fontsize=8.0)
    sns.despine(ax=ax_c)

    # Panel D: Drug x Master Regulator Reversal Heatmap (Spans full bottom row gs[1, :])
    ax_d = fig.add_subplot(gs[1, :])
    add_panel_label(ax_d, "D", x=-0.015, y=1.05)

    top14_drugs = [d for d in drug_df.head(14)["drug"] if d in drug_reg.columns]
    key_regulons = [
        "MYC", "SOX10", "E2F1", "E2F4", "FOXM1", "HALLMARK_WNT_MYC_EXCL", "NFE2L2",
        "SMAD3", "HALLMARK_TGFB_MYOCAF", "TWIST1", "STAT3", "HIF1A", "HALLMARK_SPP1_TREM2_TAM",
        "STAT1", "IRF1", "CIITA", "TBX21", "RUNX3", "IRF8", "HALLMARK_CD8_CYTOTOXIC", "HALLMARK_ANTIGEN_PRES"
    ]
    key_regulons = [r for r in key_regulons if r in drug_reg.index]
    hm_mat = drug_reg.loc[key_regulons, top14_drugs].T
    hm_mat.columns = [str(c).replace("HALLMARK_", "HM_") for c in hm_mat.columns]

    def _truncate_words(s, limit):
        s = str(s)
        if len(s) <= limit:
            return s
        cut = s[:limit]
        return cut[:cut.rfind(" ")].rstrip(" (,") if " " in cut else cut

    pretty_rows = []
    for d in top14_drugs:
        r_info = drug_df[drug_df["drug"] == d].iloc[0]
        rk = int(r_info["scRegPharm_rank"])
        moa_short = _truncate_words(str(r_info["moa"]).replace(" inhibitor", "i"), 24)
        pretty_rows.append(f"#{rk} {str(d).upper()} ({moa_short})")
    hm_mat.index = pretty_rows

    hm_vlim = max(2.2, float(np.nanmax(np.abs(hm_mat.values))) * 1.05)
    sns.heatmap(
        hm_mat, ax=ax_d, cmap="RdBu_r", vmin=-hm_vlim, vmax=hm_vlim, center=0,
        annot=True, fmt=".2f", linewidths=0.8, linecolor="white",
        cbar_kws={"label": "A375 L1000 Regulon Perturbation Z-Score", "pad": 0.02},
        annot_kws={"size": 7.6, "weight": "bold"}
    )
    n_res_cols = sum(1 for r in key_regulons if r not in ["STAT1", "IRF1", "CIITA", "TBX21", "RUNX3", "IRF8", "HALLMARK_CD8_CYTOTOXIC", "HALLMARK_ANTIGEN_PRES"])
    ax_d.axvline(n_res_cols, color="#1A252F", lw=2.2, ls="--")

    ax_d.set_xlabel("")
    ax_d.set_ylabel("")
    ax_d.set_xticklabels(hm_mat.columns, rotation=26, ha="right", fontsize=8.4, fontweight="bold")
    ax_d.set_yticklabels(hm_mat.index, rotation=0, fontsize=8.5, fontweight="bold")
    ax_d.set_title("Mechanistic Reversal of Melanoma Resistance Regulons across A375 LINCS L1000 Perturbations", fontsize=11.0, fontweight="bold", pad=30)

    fig.savefig(os.path.join(FIG_DIR, "Figure5_Drug_Repurposing_CRISPR.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "Figure5_Drug_Repurposing_CRISPR.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 5.")


def generate_figure6_clinical_validation():
    """
    Figure 6: Multi-Cohort Clinical Prognostic Stratification & Immunotherapy Response Prediction
    across 5 CRI iAtlas ICB cohorts (one pre-treatment sample per patient).
    All panels plotted from results/clinical_validation_summary.csv and *_scored_clinical.csv.
    """
    print("Generating Figure 6 (5-Cohort Clinical Validation)...")
    clin_summary = pd.read_csv(os.path.join(RES_DIR, "clinical_validation_summary.csv"))
    n_by_cohort = dict(zip(clin_summary["cohort"], clin_summary["n_patients"]))
    n_total = int(clin_summary["n_patients"].sum())

    cohorts_info = [
        ("GSE91061_Riaz_Melanoma", f"GSE91061 (Riaz Melanoma Anti-PD-1, n = {n_by_cohort['GSE91061_Riaz_Melanoma']})", "A"),
        ("PRJEB23709_Gide_Melanoma", f"PRJEB23709 (Gide Melanoma Anti-PD-1/CTLA-4, n = {n_by_cohort['PRJEB23709_Gide_Melanoma']})", "B"),
        ("DFCI_Liu_Melanoma", f"DFCI Nature Med (Liu Melanoma Anti-PD-1, n = {n_by_cohort['DFCI_Liu_Melanoma']})", "C"),
        ("IMvigor210_Urothelial_ICB", f"IMvigor210 (mUC Anti-PD-L1 Phase II, n = {n_by_cohort['IMvigor210_Urothelial_ICB']})", "D"),
    ]

    fig = plt.figure(figsize=(17.2, 11.2), dpi=300)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0, 1.12], hspace=0.36, wspace=0.30)

    km_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx_c, (c_name, c_title, p_label) in enumerate(cohorts_info):
        r_idx, c_idx = km_positions[idx_c]
        ax = fig.add_subplot(gs[r_idx, c_idx])
        add_panel_label(ax, p_label, x=-0.16, y=1.10)

        df_c = pd.read_csv(os.path.join(RES_DIR, f"{c_name}_scored_clinical.csv"), index_col=0)
        row_stat = clin_summary[clin_summary["cohort"] == c_name].iloc[0]

        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        high_mask = df_c["risk_group"] == "High-Risk (scRegPharm+)"
        low_mask = ~high_mask

        kmf_low.fit(
            df_c.loc[low_mask, "OS_months"],
            event_observed=df_c.loc[low_mask, "OS_event"],
            label=f"Low-Risk / scRegPharm- (n={low_mask.sum()})"
        )
        kmf_high.fit(
            df_c.loc[high_mask, "OS_months"],
            event_observed=df_c.loc[high_mask, "OS_event"],
            label=f"High-Risk / scRegPharm+ (n={high_mask.sum()})"
        )

        kmf_low.plot_survival_function(ax=ax, ci_show=True, color="#1F77B4", lw=2.4)
        kmf_high.plot_survival_function(ax=ax, ci_show=True, color="#D62728", lw=2.4)

        hr = row_stat["Cox_HR_per_SD"]
        ci_l = row_stat["Cox_CI_95_low"]
        ci_h = row_stat["Cox_CI_95_high"]
        p_lr = row_stat["Logrank_p_value"]
        p_str = f"P = {p_lr:.2e}" if p_lr < 0.001 else f"P = {p_lr:.4f}"

        ax.text(
            0.03, 0.05,
            f"Log-rank {p_str}\nHR = {hr:.2f} ({ci_l:.2f}–{ci_h:.2f})\nMedian OS {row_stat['Median_OS_HighRisk']:.1f} vs {row_stat['Median_OS_LowRisk']:.1f} mo",
            transform=ax.transAxes, fontsize=7.3, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#F8F9FA", edgecolor="#BDC3C7", alpha=0.94)
        )

        ax.set_xlabel("Overall Survival Time (Months)", fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Overall Survival Probability", fontsize=9.5, fontweight="bold")
        ax.set_title(c_title, fontsize=10.2, fontweight="bold", pad=8)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", frameon=True, fontsize=7.8)
        sns.despine(ax=ax)

    # Panel E (Top-Right): Multi-Cohort Predictive AUROC Comparison across All 5 Cohorts
    ax_e = fig.add_subplot(gs[0, 2])
    add_panel_label(ax_e, "E", x=-0.14, y=1.08)

    short_names_c = {
        "GSE91061_Riaz_Melanoma": "GSE91061",
        "PRJEB23709_Gide_Melanoma": "PRJEB23709",
        "DFCI_Liu_Melanoma": "DFCI Liu",
        "GSE78220_Hugo_Melanoma": "GSE78220",
        "IMvigor210_Urothelial_ICB": "IMvigor210",
    }
    short_map = {
        c: f"{short_names_c.get(c, c)}\n(n={n_by_cohort[c]})" for c in clin_summary["cohort"]
    }
    short_cohorts = [short_map.get(c, c) for c in clin_summary["cohort"]]
    x_c = np.arange(len(short_cohorts))
    w = 0.18

    # Plot the CSV score columns directly
    auc_ours_vals = clin_summary["AUC_scRegPharm"].values
    auc_tide_vals = clin_summary["AUC_TIDE"].values
    auc_deg_vals = clin_summary["AUC_Bulk_DEG"].values
    auc_ifng_vals = clin_summary["AUC_IFNG_Sig"].values

    sig_spec = [
        (auc_ours_vals, "scRegPharm Risk (Ours)", "#D62728"),
        (auc_tide_vals, "TIDE (CRI iAtlas)", "#1F77B4"),
        (auc_deg_vals, "Bulk-DEG Signature", "#2CA02C"),
        (auc_ifng_vals, "IFN-γ / T-Cell Inflamed", "#7F7F7F"),
    ]

    for idx_s, (vals, s_label, col_hex) in enumerate(sig_spec):
        offset = (idx_s - 1.5) * w
        rects = ax_e.bar(x_c + offset, vals, w, color=col_hex, edgecolor="black", lw=0.7, label=s_label)
        if idx_s == 0:
            for r in rects:
                ax_e.text(
                    r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                    f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#922B21"
                )

    ax_e.axhline(0.5, color="#7F8C8D", ls="--", lw=1.0)
    ax_e.set_xticks(x_c)
    ax_e.set_xticklabels(short_cohorts, fontsize=8.0, fontweight="bold")
    ax_e.set_ylabel("Resistance Prediction AUROC", fontsize=9.5, fontweight="bold")
    ax_e.set_title("AUROC across five ICB cohorts", fontsize=10.5, fontweight="bold", pad=8)
    ax_e.set_ylim(0.40, 1.05)
    ax_e.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_e.legend(loc="upper left", bbox_to_anchor=(1.015, 1.0), frameon=True, fontsize=7.2)
    sns.despine(ax=ax_e)

    # Panel F (Bottom-Right): RECIST v1.1 Clinical Response Breakdown across All Patients
    ax_f = fig.add_subplot(gs[1, 2])
    add_panel_label(ax_f, "F", x=-0.14, y=1.08)

    all_clin = pd.read_csv(os.path.join(RES_DIR, "all_patients_combined.csv"), index_col=0)
    recist_col = "RECIST_response" if "RECIST_response" in all_clin.columns else ("RECIST" if "RECIST" in all_clin.columns else "RECIST_category")
    recist_map = {
        "Complete Response": "CR",
        "Partial Response": "PR",
        "Stable Disease": "SD",
        "Progressive Disease": "PD",
        "CR": "CR",
        "PR": "PR",
        "SD": "SD",
        "PD": "PD",
        "Responder": "PR",
        "Non_Responder": "PD",
    }
    all_clin["RECIST_short"] = all_clin[recist_col].map(recist_map).fillna("PD")
    recist_tab = pd.crosstab(all_clin["risk_group"], all_clin["RECIST_short"], normalize="index") * 100.0
    recist_order = [r for r in ["CR", "PR", "SD", "PD"] if r in recist_tab.columns]
    recist_tab = recist_tab.loc[["Low-Risk (scRegPharm-)", "High-Risk (scRegPharm+)"], recist_order]

    recist_colors = {
        "CR": "#1F77B4",
        "PR": "#5DADE2",
        "Responder": "#2980B9",
        "SD": "#F39C12",
        "PD": "#C0392B",
        "Non_Responder": "#922B21",
    }
    recist_labels = {
        "CR": "Complete Response (CR)",
        "PR": "Partial Response (PR)",
        "Responder": "Objective Responder",
        "SD": "Stable Disease (SD)",
        "PD": "Progressive Disease (PD)",
        "Non_Responder": "Progressive / Non-Responder",
    }

    x_g = np.arange(2)
    bottom_vals = np.zeros(2)
    for r_cat in recist_order:
        vals = recist_tab[r_cat].values
        ax_f.bar(
            x_g, vals, width=0.52, bottom=bottom_vals,
            color=recist_colors.get(r_cat, "#7F8C8D"), edgecolor="black", lw=0.8, label=recist_labels.get(r_cat, r_cat)
        )
        for i, v in enumerate(vals):
            if v > 6.5:
                ax_f.text(
                    x_g[i], bottom_vals[i] + v / 2.0, f"{v:.1f}%",
                    ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color="white" if r_cat in ["CR", "PD", "Non_Responder"] else "#1A252F"
                )
        bottom_vals += vals

    nr_col = "ICB_non_responder" if "ICB_non_responder" in all_clin.columns else "icb_non_responder"
    ct_raw = pd.crosstab(all_clin["risk_group"], all_clin[nr_col])
    _, p_chi2, _, _ = stats.chi2_contingency(ct_raw)
    n_low = int((all_clin["risk_group"] == "Low-Risk (scRegPharm-)").sum())
    n_high = int((all_clin["risk_group"] == "High-Risk (scRegPharm+)").sum())

    ax_f.set_xticks(x_g)
    ax_f.set_xticklabels([f"Low-Risk\n(scRegPharm-, n={n_low})", f"High-Risk\n(scRegPharm+, n={n_high})"], fontsize=9.0, fontweight="bold")
    ax_f.set_ylabel("Proportion of Patients (%)", fontsize=9.5, fontweight="bold")
    ax_f.set_title(f"Clinical Response Stratification (N = {len(all_clin)}, P = {p_chi2:.1e})", fontsize=10.2, fontweight="bold", pad=8)
    ax_f.set_ylim(0, 108)
    ax_f.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, fontsize=7.3)
    sns.despine(ax=ax_f)

    fig.savefig(os.path.join(FIG_DIR, "Figure6_Clinical_Cohort_Validation.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "Figure6_Clinical_Cohort_Validation.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 6.")


if __name__ == "__main__":
    generate_figure4_spatial()
    generate_figure5_drug_repurposing()
    generate_figure6_clinical_validation()
    print("Figures 4, 5, and 6 regenerated successfully.")
