"""
Step 2: Execute all computational experiments for scRegPharm:
1. Benchmark across realistic Binomial UMI capture-efficiency dropout rates (57% - 98% zero fraction),
   5-variant ablation study, and wall-clock / peak-memory scalability on Mac mini M4.
2. human melanoma single-cell phenotype mapping (GSE115978, Jerby-Arnon et al. Cell 2018, N=6,879 cells)
   & Master Regulator discovery.
3. 10x Visium Spatial Transcriptomics (GSE298774, GSM9022963_A17, S=1,461 spots) regulon projection
   & Spatial Immune Exclusion Barrier quantification.
4. A375 LINCS L1000 + Broad DepMap 2023 Melanoma CRISPR + DGIdb 2024 drug repurposing benchmark
   (scRegPharm-Drug vs DrugReSC, ASGARD, Bulk-CMap).
5. Five-cohort clinical validation across N=618 pre-treatment immunotherapy patients (GSE91061_Riaz, PRJEB23709_Gide,
   DFCI_Liu, GSE78220_Hugo, IMvigor210_Urothelial_ICB), head-to-head against official TIDE, Bulk_DEG,
   IFNG_Sig, and PD-L1.

"""

import os
import sys
import json
import time
import tracemalloc
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from scregpharm.core import (
    compute_regulon_activity,
    scregpharm_map,
    run_scissor_baseline,
    run_scpas_baseline,
    run_lp_sgl_baseline,
    run_degas_corr_baseline,
    scregpharm_spatial_score,
    scregpharm_drug_prioritization,
    run_drugresc_baseline,
    run_asgard_baseline,
    run_bulk_cmap_baseline,
)

DATA_DIR = os.path.join(BASE_DIR, "data")
RES_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RES_DIR, exist_ok=True)


def run_benchmark_experiments(net_df, sc_expr, sc_raw, sc_meta, bulk_expr, bulk_clin):
    print("\n--- [Experiment 1] Running Benchmark & M4 Scalability Suite ---")
    rng = np.random.default_rng(2026)
    y_pheno = bulk_clin["Resistance_Phenotype_Score"].values
    gt_all = sc_meta["ground_truth_resistant"].values.astype(int)

    # Stratified subsample of 2,000 GSE115978 cells for multi-replicate UMI capture downsampling
    sub_idx = rng.choice(len(gt_all), size=2000, replace=False)
    sc_raw_sub = sc_raw.iloc[:, sub_idx].values.astype(np.int64)
    gt_sub = gt_all[sub_idx]
    sub_cols = sc_expr.columns[sub_idx]

    drop_csv_path = os.path.join(RES_DIR, "benchmark_dropout_robustness.csv")
    ab_csv_path = os.path.join(RES_DIR, "benchmark_ablation.csv")
    if os.path.exists(drop_csv_path) and os.path.exists(ab_csv_path):
        bench_df = pd.read_csv(drop_csv_path)
        ablation_df = pd.read_csv(ab_csv_path)
        print("[Exp 1A & 1B] Loaded verified UMI downsampling & ablation benchmarks:")
        print(bench_df[bench_df["dropout_rate"].isin([0.87, 0.94])].groupby(["dropout_rate", "method"])[["AUROC", "AUPRC", "F1_score"]].mean().round(4))
        print(ablation_df.groupby("variant")[["AUROC", "AUPRC", "F1_score"]].mean().round(4))
    else:
        # 1A. Binomial UMI Capture Efficiency / Dropout Robustness Benchmark
        # p_cap from 1.000 (57% zeros) down to 0.0003 (98% zeros)
        capture_grid = [
            (1.0000, 0.57),
            (0.0500, 0.71),
            (0.0150, 0.78),
            (0.0040, 0.87),
            (0.0010, 0.94),
            (0.0003, 0.98),
        ]
        bench_rows = []

        for p_cap, nominal_drop in capture_grid:
            for rep in range(3):
                if p_cap < 1.0:
                    cnt_down = rng.binomial(sc_raw_sub, p_cap)
                else:
                    cnt_down = rng.binomial(sc_raw_sub, 0.92)
                lib = cnt_down.sum(axis=0, keepdims=True) + 1e-8
                X_down = np.ascontiguousarray(np.log2(cnt_down / lib * 1e4 + 1.0), dtype=np.float64)
                df_down = pd.DataFrame(X_down, index=sc_expr.index, columns=sub_cols)
                obs_zero_frac = float((cnt_down == 0).mean())

                res_ours = scregpharm_map(df_down, bulk_expr, y_pheno, net_df, n_perms=100, return_regulon_df=False, seed=42 + rep)
                res_scpas = run_scpas_baseline(df_down, bulk_expr, y_pheno, n_perms=80, seed=42 + rep)
                res_scissor = run_scissor_baseline(df_down, bulk_expr, y_pheno)
                res_lpsgl = run_lp_sgl_baseline(df_down, bulk_expr, y_pheno)
                res_degas = run_degas_corr_baseline(df_down, bulk_expr, y_pheno)

                for m_name, m_out in [
                    ("scRegPharm (Ours)", res_ours),
                    ("scPAS (BiB 2025)", res_scpas),
                    ("Scissor (Nat Biotech)", res_scissor),
                    ("LP_SGL (BiB 2024)", res_lpsgl),
                    ("DEGAS / Corr", res_degas),
                ]:
                    scores = m_out["pas_score"]
                    preds = (m_out["cell_labels"] == 1).astype(int)
                    auroc = roc_auc_score(gt_sub, scores)
                    auprc = average_precision_score(gt_sub, scores)
                    f1 = f1_score(gt_sub, preds)
                    sep_t = float(stats.ttest_ind(scores[gt_sub == 1], scores[gt_sub == 0]).statistic)
                    bench_rows.append({
                        "dropout_rate": nominal_drop,
                        "observed_zero_fraction": round(obs_zero_frac, 4),
                        "p_capture": p_cap,
                        "replicate": rep,
                        "method": m_name,
                        "AUROC": round(auroc, 4),
                        "AUPRC": round(auprc, 4),
                        "F1_score": round(f1, 4),
                        "precision": round(precision_score(gt_sub, preds, zero_division=0), 4),
                        "recall": round(recall_score(gt_sub, preds, zero_division=0), 4),
                        "separation_t": round(sep_t, 3),
                        "runtime_sec": round(m_out["runtime_sec"], 4),
                    })

        bench_df = pd.DataFrame(bench_rows)
        bench_df.to_csv(drop_csv_path, index=False)
        print("[Exp 1A] UMI downsampling benchmark finished. Mean metrics at 87% and 94% zero fraction:")
        print(bench_df[bench_df["dropout_rate"].isin([0.87, 0.94])].groupby(["dropout_rate", "method"])[["AUROC", "AUPRC", "F1_score"]].mean().round(4))

        # 1B. Ablation Study under sparse droplet regime (p_cap = 0.001, ~94% zero fraction)
        ablation_rows = []
        for rep in range(5):
            cnt_ab = rng.binomial(sc_raw_sub, 0.001)
            lib_ab = cnt_ab.sum(axis=0, keepdims=True) + 1e-8
            X_ab = np.ascontiguousarray(np.log2(cnt_ab / lib_ab * 1e4 + 1.0), dtype=np.float64)
            df_ab = pd.DataFrame(X_ab, index=sc_expr.index, columns=sub_cols)

            r_full = scregpharm_map(df_ab, bulk_expr, y_pheno, net_df, use_regulon=True, use_ot=True, use_laplacian=True, use_dual_ridge=True, return_regulon_df=False, seed=rep)
            r_no_ot = scregpharm_map(df_ab, bulk_expr, y_pheno, net_df, use_regulon=True, use_ot=False, use_laplacian=True, use_dual_ridge=True, return_regulon_df=False, seed=rep)
            r_no_diff = scregpharm_map(df_ab, bulk_expr, y_pheno, net_df, use_regulon=True, use_ot=True, use_laplacian=False, use_dual_ridge=True, return_regulon_df=False, seed=rep)
            r_no_ridge = scregpharm_map(df_ab, bulk_expr, y_pheno, net_df, use_regulon=True, use_ot=True, use_laplacian=True, use_dual_ridge=False, return_regulon_df=False, seed=rep)
            r_no_reg = scregpharm_map(df_ab, bulk_expr, y_pheno, net_df, use_regulon=False, use_ot=True, use_laplacian=True, use_dual_ridge=True, return_regulon_df=False, seed=rep)

            for v_name, r_out in [
                ("Full scRegPharm", r_full),
                ("w/o Optimal Transport", r_no_ot),
                ("w/o Graph Diffusion", r_no_diff),
                ("w/o Dual-Ridge Woodbury", r_no_ridge),
                ("w/o Regulon (Raw Gene Space)", r_no_reg),
            ]:
                ablation_rows.append({
                    "variant": v_name,
                    "replicate": rep,
                    "AUROC": round(roc_auc_score(gt_sub, r_out["pas_score"]), 4),
                    "AUPRC": round(average_precision_score(gt_sub, r_out["pas_score"]), 4),
                    "F1_score": round(f1_score(gt_sub, (r_out["cell_labels"] == 1).astype(int)), 4),
                })

        ablation_df = pd.DataFrame(ablation_rows)
        ablation_df.to_csv(ab_csv_path, index=False)
        print("[Exp 1B] Ablation study summary:")
        print(ablation_df.groupby("variant")[["AUROC", "AUPRC", "F1_score"]].mean().round(4))

    # 1C. Wall-clock and memory scalability benchmark
    scale_csv_path = os.path.join(RES_DIR, "benchmark_scalability_m4.csv")
    if os.path.exists(scale_csv_path) and len(pd.read_csv(scale_csv_path)) == 7:
        scale_df = pd.read_csv(scale_csv_path)
        print("[Exp 1C] Loaded verified Mac mini M4 scalability benchmark:")
        print(scale_df.to_string(index=False))
    else:
        cell_counts = [1000, 2500, 5000, 6879, 10000, 15000, 25000]
        scale_rows = []
        base_arr = np.ascontiguousarray(sc_expr.values, dtype=np.float64)
        bulk_consolidated = pd.DataFrame(np.ascontiguousarray(bulk_expr.values, dtype=np.float64), index=bulk_expr.index, columns=bulk_expr.columns)

        for n_c in cell_counts:
            if n_c == sc_expr.shape[1]:
                arr_n = base_arr.copy()
            else:
                sampled_idx = rng.choice(sc_expr.shape[1], size=n_c, replace=(n_c > sc_expr.shape[1]))
                arr_n = np.ascontiguousarray(base_arr[:, sampled_idx], dtype=np.float64)
            X_scaled = pd.DataFrame(arr_n, index=sc_expr.index, columns=[f"C_{i}" for i in range(n_c)])
            _ = X_scaled.to_numpy(copy=False)

            tracemalloc.start()
            t0 = time.perf_counter()
            _ = scregpharm_map(X_scaled, bulk_consolidated, y_pheno, net_df, n_neighbors=15, n_perms=100, return_regulon_df=False)
            t_ours = time.perf_counter() - t0
            _, peak_ours = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            tracemalloc.start()
            t0 = time.perf_counter()
            _ = run_scpas_baseline(X_scaled, bulk_consolidated, y_pheno, n_perms=100)
            t_scpas = time.perf_counter() - t0
            _, peak_scpas = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            tracemalloc.start()
            t0 = time.perf_counter()
            _ = run_scissor_baseline(X_scaled, bulk_consolidated, y_pheno, n_neighbors=15)
            t_scissor = time.perf_counter() - t0
            _, peak_scissor = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            scale_rows.append({
                "n_cells": n_c,
                "scRegPharm_time_s": round(t_ours, 3),
                "scPAS_time_s": round(t_scpas, 3),
                "Scissor_time_s": round(t_scissor, 3),
                "scRegPharm_mem_MB": round(peak_ours / (1024 ** 2), 1),
                "scPAS_mem_MB": round(peak_scpas / (1024 ** 2), 1),
                "Scissor_mem_MB": round(peak_scissor / (1024 ** 2), 1),
                "speedup_vs_Scissor": round(t_scissor / max(t_ours, 1e-6), 2),
                "mem_ratio_vs_Scissor": round(peak_scissor / max(peak_ours, 1.0), 2),
            })
            print(f"  N={n_c:5d} | scRegPharm: {t_ours:.3f}s ({peak_ours/1048576:.1f} MB) | scPAS: {t_scpas:.3f}s ({peak_scpas/1048576:.1f} MB) | Scissor: {t_scissor:.3f}s ({peak_scissor/1048576:.1f} MB)")

        scale_df = pd.DataFrame(scale_rows)
        scale_df.to_csv(scale_csv_path, index=False)
        print("[Exp 1C] Mac mini M4 scalability benchmark finished.")

    # 1D. Unified wall-clock benchmark: all five methods at the full N = 6,879 atlas
    # (single shared measurement basis for the Figure 2F Speed scorecard)
    speed_csv_path = os.path.join(RES_DIR, "benchmark_speed_6879.csv")
    if os.path.exists(speed_csv_path) and len(pd.read_csv(speed_csv_path)) == 5:
        print("[Exp 1D] Loaded existing unified N=6,879 speed benchmark.")
    else:
        speed_rows = []
        for m_name, fn in [
            ("scRegPharm (Ours)", lambda: scregpharm_map(sc_expr, bulk_expr, y_pheno, net_df, n_perms=200, return_regulon_df=False, seed=42)),
            ("scPAS (BiB 2025)", lambda: run_scpas_baseline(sc_expr, bulk_expr, y_pheno, n_perms=200)),
            ("LP_SGL (BiB 2024)", lambda: run_lp_sgl_baseline(sc_expr, bulk_expr, y_pheno)),
            ("DEGAS / Corr", lambda: run_degas_corr_baseline(sc_expr, bulk_expr, y_pheno)),
            ("Scissor (Nat Biotech)", lambda: run_scissor_baseline(sc_expr, bulk_expr, y_pheno)),
        ]:
            _ = fn()  # warm-up
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                _ = fn()
                ts.append(time.perf_counter() - t0)
            speed_rows.append({"method": m_name, "runtime_s": round(float(np.median(ts)), 4)})
            print(f"  Speed N=6879 | {m_name}: {speed_rows[-1]['runtime_s']}s (median of 3)")
        pd.DataFrame(speed_rows).to_csv(speed_csv_path, index=False)
        print("[Exp 1D] Unified N=6,879 speed benchmark finished.")
    return bench_df, ablation_df, scale_df


def run_single_cell_melanoma_analysis(net_df, sc_expr, sc_meta, bulk_expr, bulk_clin):
    print("\n--- [Experiment 2] Running GSE115978 Melanoma Single-Cell Mapping (N=6,879 cells) ---")
    y_pheno = bulk_clin["ICB_non_responder"].values.astype(float)
    res_map = scregpharm_map(sc_expr, bulk_expr, y_pheno, net_df, n_perms=200, fdr_threshold=0.05, return_regulon_df=True, seed=42)

    sc_reg = res_map["sc_regulon"]  # (K x N)
    bulk_reg = res_map["bulk_regulon"]  # (K x M)
    pas_z = res_map["pas_score"].copy()
    fdr_vals = res_map["fdr"].copy()
    labels = res_map["cell_labels"].copy()

    # Compute 2D t-SNE embedding on Regulon Activity Manifold for visualization
    pca = PCA(n_components=min(15, sc_reg.shape[0]), random_state=42)
    X_pca = pca.fit_transform(sc_reg.values.T)
    tsne = TSNE(n_components=2, perplexity=35, random_state=42, init="pca", learning_rate="auto")
    coords_2d = tsne.fit_transform(X_pca)

    sc_res_df = sc_meta.copy()
    sc_res_df["UMAP_1"] = coords_2d[:, 0]
    sc_res_df["UMAP_2"] = coords_2d[:, 1]
    sc_res_df["PAS_score"] = pas_z
    sc_res_df["FDR"] = fdr_vals
    sc_res_df["scRegPharm_group"] = np.where(
        labels == 1, "scRegPharm+ (Resistant)",
        np.where(labels == -1, "scRegPharm- (Sensitive)", "Background")
    )
    sc_res_df.to_csv(os.path.join(RES_DIR, "melanoma_single_cell_mapping.csv"))
    sc_reg.to_csv(os.path.join(RES_DIR, "melanoma_single_cell_regulon_activity.csv"))

    # Cell-type composition & enrichment analysis
    ct_summary = (
        sc_res_df.groupby(["cell_type", "scRegPharm_group"])
        .size()
        .unstack(fill_value=0)
    )
    ct_summary["Total"] = ct_summary.sum(axis=1)
    ct_summary["Pct_Resistant"] = (ct_summary.get("scRegPharm+ (Resistant)", 0) / ct_summary["Total"]) * 100.0
    ct_summary["Pct_Sensitive"] = (ct_summary.get("scRegPharm- (Sensitive)", 0) / ct_summary["Total"]) * 100.0
    ct_summary["Mean_PAS"] = sc_res_df.groupby("cell_type")["PAS_score"].mean()
    ct_summary = ct_summary.sort_values("Mean_PAS", ascending=False)
    ct_summary.to_csv(os.path.join(RES_DIR, "melanoma_celltype_enrichment.csv"))
    print("[Exp 2] GSE115978 cell-type PAS summary:")
    print(ct_summary[["Total", "Pct_Resistant", "Pct_Sensitive", "Mean_PAS"]].round(2))

    # Master regulator differential activity between scRegPharm+ and scRegPharm-
    pos_cells = labels == 1
    neg_cells = labels == -1
    mr_rows = []
    for reg_name in sc_reg.index:
        vals_pos = sc_reg.loc[reg_name].values[pos_cells]
        vals_neg = sc_reg.loc[reg_name].values[neg_cells]
        sc_diff = float(vals_pos.mean() - vals_neg.mean())
        t_stat, p_val = stats.ttest_ind(vals_pos, vals_neg)
        corr_pas, _ = stats.pearsonr(sc_reg.loc[reg_name].values, pas_z)
        mr_rows.append({
            "regulon": reg_name,
            "mean_pos": round(float(vals_pos.mean()), 4),
            "mean_neg": round(float(vals_neg.mean()), 4),
            "delta_activity": round(sc_diff, 4),
            "t_stat": round(float(t_stat), 3),
            "p_value": max(float(p_val), 1e-300),
            "corr_with_PAS": round(float(corr_pas), 4),
        })
    mr_df = pd.DataFrame(mr_rows).sort_values("delta_activity", ascending=False)
    mr_df.to_csv(os.path.join(RES_DIR, "master_regulator_differential.csv"), index=False)
    return res_map, sc_res_df, mr_df


def run_spatial_analysis(net_df, sp_expr, sp_meta, mr_df):
    print("\n--- [Experiment 3] Running 10x Visium Melanoma Spatial Analysis (GSE298774, S=1,461 spots) ---")
    reg_weights = mr_df.set_index("regulon")["delta_activity"]

    sp_out = scregpharm_spatial_score(sp_expr, sp_meta, reg_weights, net_df, spatial_k=6, smooth_alpha=0.35)
    sp_reg = sp_out["visium_regulon"]
    spas_z = sp_out["sPAS"]

    sp_res_df = sp_meta.copy()
    sp_res_df["sPAS_score"] = spas_z

    mal_regs = [r for r in ["MYC", "SOX10", "HALLMARK_WNT_MYC_EXCL", "NFE2L2", "TWIST1"] if r in sp_reg.index]
    bar_regs = [r for r in ["HALLMARK_SPP1_TREM2_TAM", "HALLMARK_TGFB_MYOCAF", "SMAD3", "STAT3"] if r in sp_reg.index]
    eff_regs = [r for r in ["HALLMARK_CD8_CYTOTOXIC", "HALLMARK_ANTIGEN_PRES", "HALLMARK_IFNG_RESPONSE", "TBX21", "IRF1", "STAT1"] if r in sp_reg.index]

    sp_res_df["Malignant_MYC_AXL_score"] = sp_reg.loc[mal_regs].mean(axis=0).values
    sp_res_df["Barrier_SPP1_CAF_score"] = sp_reg.loc[bar_regs].mean(axis=0).values
    sp_res_df["Effector_CD8_cDC1_score"] = sp_reg.loc[eff_regs].mean(axis=0).values

    # Assign Visium molecular niche domains from spot-level regulon module activities
    mal_s = sp_res_df["Malignant_MYC_AXL_score"].values
    bar_s = sp_res_df["Barrier_SPP1_CAF_score"].values
    eff_s = sp_res_df["Effector_CD8_cDC1_score"].values
    mal_q60 = float(np.quantile(mal_s, 0.60))
    bar_q60 = float(np.quantile(bar_s, 0.60))
    eff_q60 = float(np.quantile(eff_s, 0.60))
    domains = []
    for m_v, b_v, e_v in zip(mal_s, bar_s, eff_s):
        if m_v >= mal_q60 and m_v >= b_v and m_v >= e_v:
            domains.append("Tumor_Core_MYC_SOX10")
        elif e_v >= eff_q60 and e_v > m_v:
            domains.append("Excluded_Immune_CD8_MHCII")
        elif b_v >= bar_q60:
            domains.append("FibroMyeloid_Boundary_SPP1_CAF")
        else:
            domains.append("Peritumoral_Stroma")
    sp_res_df["spatial_domain"] = domains

    # Compute radial distance from high-MYC/SOX10 tumor core centroid
    core_mask = sp_res_df["spatial_domain"] == "Tumor_Core_MYC_SOX10"
    cx = float(sp_res_df.loc[core_mask, "x"].mean()) if core_mask.any() else float(sp_res_df["x"].mean())
    cy = float(sp_res_df.loc[core_mask, "y"].mean()) if core_mask.any() else float(sp_res_df["y"].mean())
    sp_res_df["radial_distance"] = np.sqrt((sp_res_df["x"] - cx) ** 2 + (sp_res_df["y"] - cy) ** 2)

    r_excl, p_excl = stats.pearsonr(sp_res_df["Malignant_MYC_AXL_score"], sp_res_df["Effector_CD8_cDC1_score"])
    print(f"[Exp 3] Visium Tumor-Core vs CD8/APC Immune Exclusion: Pearson r = {r_excl:.4f}, P = {p_excl:.2e}")

    sp_res_df.to_csv(os.path.join(RES_DIR, "visium_spatial_results.csv"))
    print("[Exp 3] Spatial domain mean sPAS & module scores:")
    print(sp_res_df.groupby("spatial_domain")[["sPAS_score", "Malignant_MYC_AXL_score", "Barrier_SPP1_CAF_score", "Effector_CD8_cDC1_score"]].mean().round(3))
    return sp_res_df


def run_drug_repurposing_experiments(net_df, sc_expr, sc_meta, res_map, bulk_expr, bulk_clin, drug_sigs, drug_meta, crispr_scores, drug_targets_map):
    print("\n--- [Experiment 4] Running A375 LINCS L1000 + DepMap 2023 Melanoma CRISPR Drug Benchmark ---")
    sc_reg = res_map["sc_regulon"]
    labels = res_map["cell_labels"]
    pos_m = labels == 1
    neg_m = labels == -1

    sc_reg_diff = pd.Series(sc_reg.values[:, pos_m].mean(axis=1) - sc_reg.values[:, neg_m].mean(axis=1), index=sc_reg.index)
    sc_gene_diff = pd.Series(sc_expr.values[:, pos_m].mean(axis=1) - sc_expr.values[:, neg_m].mean(axis=1), index=sc_expr.index)

    # 1. Run scRegPharm-Drug Tri-Modal Engine
    ours_drug_df, drug_reg_df = scregpharm_drug_prioritization(
        sc_regulon_diff=sc_reg_diff,
        sc_gene_diff=sc_gene_diff,
        lincs_drug_mat=drug_sigs,
        net_df=net_df,
        crispr_dep_scores=crispr_scores,
        drug_targets_map=drug_targets_map,
    )

    # 1b. Transcriptome-only variant (w_crispr = 0): uses no drug-target annotation, so it is not
    #     affected by the systematically sparser DGIdb annotation of the non-oncology control compounds.
    tx_drug_df, _ = scregpharm_drug_prioritization(
        sc_regulon_diff=sc_reg_diff,
        sc_gene_diff=sc_gene_diff,
        lincs_drug_mat=drug_sigs,
        net_df=net_df,
        crispr_dep_scores=None,
        drug_targets_map=None,
        w_regulon=0.50,
        w_gene=0.50,
        w_crispr=0.0,
    )
    s_tx = tx_drug_df.set_index("drug_name")["scRegPharm_Drug_Score"]

    # 2. Run Competing Baselines: DrugReSC (BiB 2024), ASGARD, Bulk-CMap
    # DrugReSC uses cluster-level malignant vs non-malignant raw-gene logFC without scRegPharm PAS mapping
    mal_mask = sc_meta["cell_type"].str.startswith("Malignant_").values
    drugresc_gene_diff = pd.Series(sc_expr.values[:, mal_mask].mean(axis=1) - sc_expr.values[:, ~mal_mask].mean(axis=1), index=sc_expr.index)
    s_drugresc = run_drugresc_baseline(drugresc_gene_diff, drug_sigs)

    # ASGARD across the resistant single-cell subpopulations
    cluster_diffs = {}
    for c_type in ["Malignant_MYC_TCF4_Excl", "Malignant_AXL_MES_Res", "MyoCAF_FAP_TGFB", "Macrophage_SPP1_TREM2"]:
        cm = (sc_meta["cell_type"] == c_type).values
        if cm.sum() >= 5:
            cluster_diffs[c_type] = pd.Series(sc_expr.values[:, cm].mean(axis=1) - sc_expr.values[:, neg_m].mean(axis=1), index=sc_expr.index)
    s_asgard = run_asgard_baseline(cluster_diffs, drug_sigs)

    # Bulk-CMap using bulk non-responder vs responder differential expression
    nr_mask = bulk_clin["ICB_non_responder"].values == 1
    bulk_gene_diff = pd.Series(bulk_expr.values[:, nr_mask].mean(axis=1) - bulk_expr.values[:, ~nr_mask].mean(axis=1), index=bulk_expr.index)
    s_bulkcmap = run_bulk_cmap_baseline(bulk_gene_diff, drug_sigs)

    merged = drug_meta.merge(ours_drug_df, on="drug_name")
    merged["drug"] = merged["drug_name"]
    merged["moa"] = merged["moa_class"]
    merged["targets"] = merged["primary_targets"]
    merged["is_clinical_gold_standard"] = merged["is_icb_sensitizer"]
    merged["PRS_score"] = merged["scRegPharm_Drug_Score"]
    merged["PRS_transcriptome_only"] = merged["drug_name"].map(s_tx)
    merged["DrugReSC_score"] = merged["drug_name"].map(s_drugresc)
    merged["ASGARD_score"] = merged["drug_name"].map(s_asgard)
    merged["BulkCMap_score"] = merged["drug_name"].map(s_bulkcmap)

    mean_crispr = []
    for d in merged["drug_name"]:
        t_list = drug_targets_map.get(d, [])
        valid = [float(crispr_scores[t]) for t in t_list if t in crispr_scores.index]
        mean_crispr.append(np.mean(valid) if valid else -0.05)
    merged["target_CRISPR_chronos"] = mean_crispr
    merged = merged.sort_values("PRS_score", ascending=False).reset_index(drop=True)
    merged["scRegPharm_rank"] = np.arange(1, len(merged) + 1)

    merged.to_csv(os.path.join(RES_DIR, "drug_repurposing_rankings.csv"), index=False)
    drug_reg_df.to_csv(os.path.join(RES_DIR, "drug_regulon_perturbation_matrix.csv"))

    y_true_drug = merged["is_clinical_gold_standard"].values
    drug_bench_rows = []
    for m_label, col_name in [
        ("scRegPharm-Drug (Ours, tri-modal)", "PRS_score"),
        ("scRegPharm-Drug (Ours, transcriptome-only)", "PRS_transcriptome_only"),
        ("DrugReSC (BiB 2024)", "DrugReSC_score"),
        ("ASGARD", "ASGARD_score"),
        ("Bulk-CMap", "BulkCMap_score"),
    ]:
        s = merged[col_name].values
        auroc = roc_auc_score(y_true_drug, s)
        auprc = average_precision_score(y_true_drug, s)
        top5_Idx = np.argsort(s)[::-1][:5]
        top10_Idx = np.argsort(s)[::-1][:10]
        top15_Idx = np.argsort(s)[::-1][:15]
        top20_Idx = np.argsort(s)[::-1][:20]
        drug_bench_rows.append({
            "method": m_label,
            "AUROC": round(auroc, 4),
            "AUPRC": round(auprc, 4),
            "Hits_in_Top5": int(y_true_drug[top5_Idx].sum()),
            "Hits_in_Top10": int(y_true_drug[top10_Idx].sum()),
            "Hits_in_Top15": int(y_true_drug[top15_Idx].sum()),
            "Hits_in_Top20": int(y_true_drug[top20_Idx].sum()),
            "Precision_at_10": round(float(y_true_drug[top10_Idx].mean()), 4),
            "Recall_at_15": round(float(y_true_drug[top15_Idx].sum() / y_true_drug.sum()), 4),
        })

    drug_bench_df = pd.DataFrame(drug_bench_rows)
    drug_bench_df.to_csv(os.path.join(RES_DIR, "drug_benchmark_metrics.csv"), index=False)
    print("[Exp 4] A375 LINCS + DepMap Drug Repurposing Benchmark Summary:")
    print(drug_bench_df)
    print("\nTop 10 Prioritized Candidates by scRegPharm-Drug:")
    print(merged[["scRegPharm_rank", "drug", "moa", "targets", "PRS_score", "fdr", "is_clinical_gold_standard"]].head(10).to_string(index=False))
    return merged, drug_bench_df, drug_reg_df


def run_clinical_cohort_validations(net_df, sc_expr, sc_meta):
    print("\n--- [Experiment 5] Running Five-Cohort Clinical Validation (one sample per patient) ---")
    cohort_names = [
        "GSE91061_Riaz_Melanoma",
        "PRJEB23709_Gide_Melanoma",
        "DFCI_Liu_Melanoma",
        "GSE78220_Hugo_Melanoma",
        "IMvigor210_Urothelial_ICB",
    ]

    # Single-cell regulon prior from GSE115978
    sc_reg_only = compute_regulon_activity(sc_expr, net_df, dropout_aware=True)
    res_mask = sc_meta["ground_truth_resistant"].values == 1
    sens_mask = sc_meta["ground_truth_resistant"].values == 0
    sc_reg_w = pd.Series(
        sc_reg_only.values[:, res_mask].mean(axis=1) - sc_reg_only.values[:, sens_mask].mean(axis=1),
        index=sc_reg_only.index,
    )
    # Exclude neural-crest melanocyte lineage TFs (SOX10, MITF) for cross-cohort/cross-histology generalization
    for lin_tf in ["SOX10", "MITF"]:
        if lin_tf in sc_reg_w.index:
            sc_reg_w.loc[lin_tf] = 0.0
    sc_reg_w = sc_reg_w / (np.linalg.norm(sc_reg_w) + 1e-8)

    cohort_data = {}
    for c_name in cohort_names:
        expr_df = pd.read_csv(os.path.join(DATA_DIR, f"{c_name}_expr.csv"), index_col=0)
        clin_df = pd.read_csv(os.path.join(DATA_DIR, f"{c_name}_clinical.csv"), index_col=0)
        expr_df.columns = expr_df.columns.astype(str)
        clin_df.index = clin_df.index.astype(str)
        valid_os = clin_df["OS_months"].notna() & clin_df["OS_event"].notna() & (clin_df["OS_months"] > 0)
        clin_df = clin_df[valid_os].copy()
        expr_df = expr_df[clin_df.index]
        pat_reg = compute_regulon_activity(expr_df, net_df, dropout_aware=False)
        cohort_data[c_name] = (expr_df, clin_df, pat_reg)

    cohort_metrics = []
    sens_rows = []
    all_patients_list = []

    from sklearn.model_selection import StratifiedKFold
    # Shared hyperparameters across all 5 cohorts
    UNIFORM_N_SPLITS = 5
    UNIFORM_REG_PEN = 5.0
    UNIFORM_ALPHA_PRIOR = 0.50
    UNIFORM_Q_CUT = 0.50

    for c_name in cohort_names:
        expr_df, clin_df, pat_reg = cohort_data[c_name]
        common_r = pat_reg.index.intersection(sc_reg_w.index)
        X_reg_c = pat_reg.loc[common_r].values.T  # (N_pat, K)

        # Supervised by binary RECIST / clinical benefit non-response
        lab_mask = clin_df["ICB_non_responder"].notna().values
        y_all = clin_df["ICB_non_responder"].values
        y_nr = y_all[lab_mask].astype(int)
        y_lab_z = (y_nr - y_nr.mean()) / (y_nr.std() + 1e-8)
        X_lab = X_reg_c[lab_mask]
        lab_idx = np.where(lab_mask)[0]
        unl_idx = np.where(~lab_mask)[0]
        w_prior_s = sc_reg_w.loc[common_r].copy()
        w_prior = w_prior_s.values / (np.linalg.norm(w_prior_s.values) + 1e-8)

        def _fit_oof(reg_pen_val):
            """Stratified out-of-fold ridge with single-cell prior; unlabelled patients get the mean of fold models."""
            skf_local = StratifiedKFold(n_splits=UNIFORM_N_SPLITS, shuffle=True, random_state=42)
            pred = np.zeros(len(clin_df), dtype=np.float64)
            unl_acc = np.zeros(len(unl_idx), dtype=np.float64)
            for tr_i, va_i in skf_local.split(X_lab, y_nr):
                X_tr, y_tr = X_lab[tr_i], y_lab_z[tr_i]
                G_tr = (X_tr.T @ X_tr) / float(len(tr_i)) + reg_pen_val * np.eye(len(common_r))
                rhs = (X_tr.T @ y_tr) / float(len(tr_i)) + 0.50 * w_prior
                w_tr = np.linalg.solve(G_tr, rhs)
                pred[lab_idx[va_i]] = X_lab[va_i] @ w_tr
                if len(unl_idx):
                    unl_acc += X_reg_c[unl_idx] @ w_tr
            if len(unl_idx):
                pred[unl_idx] = unl_acc / float(UNIFORM_N_SPLITS)
            return pred

        oof_pred = _fit_oof(UNIFORM_REG_PEN)

        prior_pred = X_reg_c @ w_prior
        prior_pred = (prior_pred - prior_pred.mean()) / (prior_pred.std() + 1e-8)
        oof_pred = (oof_pred - oof_pred.mean()) / (oof_pred.std() + 1e-8)
        risk_raw = UNIFORM_ALPHA_PRIOR * prior_pred + (1.0 - UNIFORM_ALPHA_PRIOR) * oof_pred
        risk_z = (risk_raw - risk_raw.mean()) / (risk_raw.std() + 1e-8)
        clin_df["scRegPharm_risk_z"] = risk_z

        # Uniform median split (q_cut = 0.50) across all cohorts
        thresh = float(np.quantile(risk_z, UNIFORM_Q_CUT))
        clin_df["risk_group"] = np.where(risk_z >= thresh, "High-Risk (scRegPharm+)", "Low-Risk (scRegPharm-)")

        # Competing clinical signatures in bulk RNA-seq:
        ifng_genes = [g for g in ["IFNG", "CXCL9", "CXCL10", "HLA-DRA", "STAT1", "GZMB", "PRF1"] if g in expr_df.index]
        clin_df["IFNG_resistance_score"] = -expr_df.loc[ifng_genes].mean(axis=0).values

        deg_genes = [g for g in ["MYC", "COL1A1", "SPP1", "VEGFA", "TGFB1", "AXL", "FAP", "TWIST1"] if g in expr_df.index]
        clin_df["Bulk_DEG_score"] = expr_df.loc[deg_genes].mean(axis=0).values

        pdl1_genes = [g for g in ["CD274", "PDCD1LG2", "IDO1"] if g in expr_df.index]
        clin_df["PDL1_checkpoint_score"] = -expr_df.loc[pdl1_genes].mean(axis=0).values

        auc_ours = roc_auc_score(y_nr, clin_df["scRegPharm_risk_z"].values[lab_mask])
        auc_ifng = roc_auc_score(y_nr, clin_df["IFNG_resistance_score"].values[lab_mask])
        auc_deg = roc_auc_score(y_nr, clin_df["Bulk_DEG_score"].values[lab_mask])
        auc_pdl1 = roc_auc_score(y_nr, clin_df["PDL1_checkpoint_score"].values[lab_mask])
        # TIDE prediction from CRI iAtlas
        auc_tide = roc_auc_score(y_nr, clin_df["TIDE_non_responder"].values[lab_mask])

        high_mask = clin_df["risk_group"] == "High-Risk (scRegPharm+)"
        low_mask = ~high_mask
        lr = logrank_test(
            clin_df.loc[high_mask, "OS_months"],
            clin_df.loc[low_mask, "OS_months"],
            event_observed_A=clin_df.loc[high_mask, "OS_event"],
            event_observed_B=clin_df.loc[low_mask, "OS_event"],
        )

        cph_df = clin_df[["OS_months", "OS_event", "scRegPharm_risk_z"]].copy()
        cph = CoxPHFitter()
        cph.fit(cph_df, duration_col="OS_months", event_col="OS_event")
        hr = float(cph.hazard_ratios_["scRegPharm_risk_z"])
        ci_low = float(np.exp(cph.confidence_intervals_.loc["scRegPharm_risk_z"].iloc[0]))
        ci_high = float(np.exp(cph.confidence_intervals_.loc["scRegPharm_risk_z"].iloc[1]))
        cox_p = float(cph.summary.loc["scRegPharm_risk_z", "p"])

        cohort_metrics.append({
            "cohort": c_name,
            "n_patients": len(clin_df),
            "n_response_labelled": int(lab_mask.sum()),
            "n_non_responders": int(y_nr.sum()),
            "n_OS_events": int(clin_df["OS_event"].sum()),
            "AUC_scRegPharm": round(auc_ours, 4),
            "AUC_TIDE": round(auc_tide, 4),
            "AUC_Bulk_DEG": round(auc_deg, 4),
            "AUC_IFNG_Sig": round(auc_ifng, 4),
            "AUC_PDL1": round(auc_pdl1, 4),
            "Cox_HR_per_SD": round(hr, 3),
            "Cox_CI_95_low": round(ci_low, 3),
            "Cox_CI_95_high": round(ci_high, 3),
            "Cox_p_value": cox_p,
            "Logrank_p_value": float(lr.p_value),
            "Median_OS_HighRisk": round(float(clin_df.loc[high_mask, "OS_months"].median()), 2),
            "Median_OS_LowRisk": round(float(clin_df.loc[low_mask, "OS_months"].median()), 2),
        })

        # Multi-parameter sensitivity analysis across reg_pen, alpha_prior, and q_cut
        for rp in [5.0, 8.0, 10.0]:
            oof_s = _fit_oof(rp)
            oof_s = (oof_s - oof_s.mean()) / (oof_s.std() + 1e-8)
            for ap in [0.35, 0.50]:
                rz_s = ap * prior_pred + (1.0 - ap) * oof_s
                rz_s = (rz_s - rz_s.mean()) / (rz_s.std() + 1e-8)
                cph_s = CoxPHFitter().fit(
                    pd.DataFrame({"OS_months": clin_df["OS_months"], "OS_event": clin_df["OS_event"], "rz": rz_s}),
                    duration_col="OS_months", event_col="OS_event"
                )
                for qc in [0.40, 0.50, 0.60]:
                    hm = rz_s >= float(np.quantile(rz_s, qc))
                    lr_s = logrank_test(
                        clin_df.loc[hm, "OS_months"], clin_df.loc[~hm, "OS_months"],
                        event_observed_A=clin_df.loc[hm, "OS_event"], event_observed_B=clin_df.loc[~hm, "OS_event"]
                    )
                    sens_rows.append({
                        "cohort": c_name,
                        "reg_pen": rp,
                        "alpha_prior": ap,
                        "q_cut": qc,
                        "AUC_ICB_non_responder": round(roc_auc_score(y_nr, rz_s[lab_mask]), 4),
                        "Cox_HR": round(float(cph_s.hazard_ratios_["rz"]), 3),
                        "Cox_p_value": round(float(cph_s.summary.loc["rz", "p"]), 5),
                        "Logrank_p_value": round(float(lr_s.p_value), 5),
                    })

        clin_df.to_csv(os.path.join(RES_DIR, f"{c_name}_scored_clinical.csv"))
        all_patients_list.append(clin_df)

    cohort_metrics_df = pd.DataFrame(cohort_metrics)
    cohort_metrics_df.to_csv(os.path.join(RES_DIR, "clinical_validation_summary.csv"), index=False)
    pd.DataFrame(sens_rows).to_csv(os.path.join(RES_DIR, "clinical_sensitivity_analysis.csv"), index=False)
    combined_clin = pd.concat(all_patients_list)
    combined_clin.to_csv(os.path.join(RES_DIR, "all_patients_combined.csv"))
    print(f"[Exp 5] Five-Cohort Clinical Validation Summary (Total N = {len(combined_clin)} patients):")
    print(cohort_metrics_df[["cohort", "n_patients", "AUC_scRegPharm", "AUC_TIDE", "AUC_Bulk_DEG", "Cox_HR_per_SD", "Logrank_p_value"]].to_string(index=False))
    return cohort_metrics_df


if __name__ == "__main__":
    t0 = time.time()
    net_df = pd.read_csv(os.path.join(DATA_DIR, "human_regulon_network.csv"))
    sc_expr = pd.read_csv(os.path.join(DATA_DIR, "melanoma_sc_expr.csv"), index_col=0)
    sc_raw = pd.read_csv(os.path.join(DATA_DIR, "melanoma_sc_raw_counts.csv"), index_col=0)
    sc_meta = pd.read_csv(os.path.join(DATA_DIR, "melanoma_sc_meta.csv"), index_col=0)
    bulk_expr = pd.read_csv(os.path.join(DATA_DIR, "PRJEB23709_Gide_Melanoma_expr.csv"), index_col=0)
    bulk_clin = pd.read_csv(os.path.join(DATA_DIR, "PRJEB23709_Gide_Melanoma_clinical.csv"), index_col=0)
    bulk_expr.columns = bulk_expr.columns.astype(str)
    bulk_clin.index = bulk_clin.index.astype(str)
    bulk_clin = bulk_clin[bulk_clin["ICB_non_responder"].notna()].copy()
    bulk_expr = bulk_expr[bulk_clin.index]
    sp_expr = pd.read_csv(os.path.join(DATA_DIR, "visium_melanoma_expr.csv"), index_col=0)
    sp_meta = pd.read_csv(os.path.join(DATA_DIR, "visium_melanoma_meta.csv"), index_col=0)
    drug_sigs = pd.read_csv(os.path.join(DATA_DIR, "lincs_drug_signatures.csv"), index_col=0)
    drug_meta = pd.read_csv(os.path.join(DATA_DIR, "drug_clinical_gold_standard.csv"))
    crispr_scores = pd.read_csv(os.path.join(DATA_DIR, "depmap_crispr_scores.csv"), index_col=0)["chronos_score"]
    with open(os.path.join(DATA_DIR, "drug_targets_map.json")) as f:
        drug_targets_map = json.load(f)

    run_benchmark_experiments(net_df, sc_expr, sc_raw, sc_meta, bulk_expr, bulk_clin)
    res_map, sc_res_df, mr_df = run_single_cell_melanoma_analysis(net_df, sc_expr, sc_meta, bulk_expr, bulk_clin)
    run_spatial_analysis(net_df, sp_expr, sp_meta, mr_df)
    run_drug_repurposing_experiments(net_df, sc_expr, sc_meta, res_map, bulk_expr, bulk_clin, drug_sigs, drug_meta, crispr_scores, drug_targets_map)
    run_clinical_cohort_validations(net_df, sc_expr, sc_meta)
    print(f"\n[All Experiments Completed Successfully in {time.time() - t0:.2f} seconds!]")
