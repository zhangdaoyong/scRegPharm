"""
external_drug_validation.py
===========================
External pharmacological validation of scRegPharm drug-reversal rankings against
independent cell-line drug-sensitivity resources (GDSC / PRISM secondary screen).

Protocol:
  * Mapping: LINCS compound names -> GDSC DRUG_NAME via normalized strings and
    the GDSC compound synonym table (GDSC1+GDSC2 union); unmapped compounds are
    reported without imputation.
  * Metric 1 (primary, drug level): Spearman rho between scRegPharm reversal
    rank and median LN_IC50 of the drug across SKCM (melanoma) cell lines.
    One-sided permutation p-value in the positive direction (1000 permutations).
  * Metric 2 (A375 single-line): Spearman between reversal rank and A375 IC50/AUC.
  * Metric 3 (panel enrichment): Mann-Whitney U comparing sensitivity (AUC /
    viability) of top-K vs remaining mapped drugs; enrichment AUROC reported.
  * GDSC1 and GDSC2 are analyzed separately and reported side by side.
"""

import re
import numpy as np
import pandas as pd
from scipy import stats as sps


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_gdsc_panel(gdsc_path: str, compounds_path: str, tcga_desc: str = "SKCM"):
    """
    Returns (panel_df, name_map) where panel_df has one row per
    (drug_name, cell_line) with LN_IC50/AUC for the requested TCGA subtype,
    and name_map maps normalized drug names -> official GDSC DRUG_NAME.
    """
    gd = pd.read_csv(gdsc_path)
    comp = pd.read_csv(compounds_path)

    name_map = {_norm(n): n for n in gd["DRUG_NAME"].unique()}
    for _, r in comp.iterrows():
        official = name_map.get(_norm(r["DRUG_NAME"]), r["DRUG_NAME"])
        for syn in str(r["SYNONYMS"]).split(","):
            k = _norm(syn)
            if k:
                name_map.setdefault(k, official)

    panel = gd[gd["TCGA_DESC"] == tcga_desc][
        ["DRUG_NAME", "CELL_LINE_NAME", "LN_IC50", "AUC", "RMSE", "Z_SCORE"]
    ].copy()
    return panel, name_map


def _perm_p_rho(rho: float, x: np.ndarray, y: np.ndarray, n_perm: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        null.append(sps.spearmanr(x, rng.permutation(y)).statistic)
    null = np.asarray(null)
    if rho >= 0:
        return float((np.sum(null >= rho) + 1) / (n_perm + 1))
    return float((np.sum(null <= rho) + 1) / (n_perm + 1))


def validate_against_gdsc(
    reversal_rank_df: pd.DataFrame,
    gdsc_panel: pd.DataFrame,
    name_map: dict,
    drug_name_col: str = "drug_name",
    rank_col: str = "scRegPharm_rank",
    top_k: int = 10,
    max_rmse_quantile: float = 0.95,
    cell_line_filter: str = None,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict:
    """
    reversal_rank_df : ranking table from scRegPharm-Drug (one row per compound).
    gdsc_panel       : from load_gdsc_panel().
    cell_line_filter : optional list of cell-line names to restrict to (e.g. melanoma).
    """
    rank_df = reversal_rank_df.copy()
    rank_df["_gdsc_name"] = rank_df[drug_name_col].map(lambda d: name_map.get(_norm(d)))

    mapped = rank_df[rank_df["_gdsc_name"].notna()].copy()
    panel = gdsc_panel
    if cell_line_filter is not None:
        panel = panel[panel["CELL_LINE_NAME"].isin(cell_line_filter)]

    # Drop poorly fitted curves (top 5% RMSE by default) before aggregation
    if "RMSE" in panel.columns and panel["RMSE"].notna().sum() > 10:
        thr = panel["RMSE"].quantile(max_rmse_quantile)
        panel = panel[panel["RMSE"] <= thr]

    med = panel.groupby("DRUG_NAME").agg(
        median_LN_IC50=("LN_IC50", "median"),
        median_AUC=("AUC", "median"),
        n_cell_lines=("CELL_LINE_NAME", "nunique"),
    )

    merged = mapped.merge(med, left_on="_gdsc_name", right_index=True, how="inner")
    n_eval = len(merged)
    result = {"n_compounds_mapped": int(mapped.shape[0]), "n_compounds_evaluated": n_eval}
    if n_eval < 8:
        result["warning"] = "too few mapped compounds for robust correlation"
        return result

    ranks = merged[rank_col].values.astype(float)
    ic50 = merged["median_LN_IC50"].values.astype(float)
    auc = merged["median_AUC"].values.astype(float)

    rho_ic50 = sps.spearmanr(ranks, ic50)
    rho_auc = sps.spearmanr(ranks, auc)
    result["spearman_rank_vs_median_IC50"] = round(float(rho_ic50.statistic), 4)
    result["pval_rank_vs_median_IC50"] = round(float(rho_ic50.pvalue), 4)
    result["perm_pval_rank_vs_median_IC50"] = round(
        _perm_p_rho(float(rho_ic50.statistic), ranks, ic50, n_perm=n_perm, seed=seed), 4
    )
    result["spearman_rank_vs_median_AUC"] = round(float(rho_auc.statistic), 4)
    result["pval_rank_vs_median_AUC"] = round(float(rho_auc.pvalue), 4)

    # Metric 2: A375 single-line sensitivity (if present in panel)
    a375 = panel[panel["CELL_LINE_NAME"].str.upper().str.contains("A375")]
    if len(a375) > 0:
        m_a375 = mapped.merge(
            a375.groupby("DRUG_NAME").agg(A375_LN_IC50=("LN_IC50", "median"), A375_AUC=("AUC", "median")),
            left_on="_gdsc_name", right_index=True, how="inner",
        )
        if len(m_a375) >= 6:
            result["n_A375_evaluated"] = len(m_a375)
            result["spearman_rank_vs_A375_IC50"] = round(
                float(sps.spearmanr(m_a375[rank_col], m_a375["A375_LN_IC50"]).statistic), 4
            )

    # Metric 3: top-K enrichment among mapped drugs
    merged_sorted = merged.sort_values(rank_col)
    top = merged_sorted.head(min(top_k, n_eval))
    rest = merged_sorted.iloc[len(top):]
    if len(rest) >= 5:
        u = sps.mannwhitneyu(top["median_AUC"], rest["median_AUC"], alternative="less")
        result["topK_vs_rest_AUC_mannwhitney_p"] = round(float(u.pvalue), 4)
        labels = np.concatenate([np.ones(len(top)), np.zeros(len(rest))])
        vals = np.concatenate([-top["median_AUC"].values, -rest["median_AUC"].values])
        from sklearn.metrics import roc_auc_score
        result["topK_enrichment_AUROC"] = round(float(roc_auc_score(labels, vals)), 4)

    return result


def run_external_validation(
    rankings_csv: str,
    gdsc1_csv: str,
    gdsc2_csv: str,
    compounds_csv: str,
    out_csv: str,
    tcga_desc: str = "SKCM",
    top_k: int = 10,
):
    rank_df = pd.read_csv(rankings_csv)
    rows = []
    for tag, gdsc_csv in (("GDSC1", gdsc1_csv), ("GDSC2", gdsc2_csv)):
        panel, name_map = load_gdsc_panel(gdsc_csv, compounds_csv, tcga_desc=tcga_desc)
        res = validate_against_gdsc(rank_df, panel, name_map, top_k=top_k)
        res = {"dataset": tag, **res}
        rows.append(res)
        print(f"  [{tag}] mapped={res.get('n_compounds_mapped')} evaluated={res.get('n_compounds_evaluated')} "
              f"rho(IC50)={res.get('spearman_rank_vs_median_IC50')} perm_p={res.get('perm_pval_rank_vs_median_IC50')}")
    out = pd.DataFrame(rows)
    # Second resource: PRISM secondary screen (Broad, dose-response IC50/AUC)
    prism_csv = os.path.join(base, "data", "raw_downloads", "prism_secondary_dose_response.csv")
    if os.path.exists(prism_csv):
        prism_row = run_prism_validation(
            rankings_csv=os.path.join(base, "results", "drug_repurposing_rankings.csv"),
            prism_csv=prism_csv,
            out_csv=os.path.join(base, "results", "external_drug_validation_prism.csv"),
        )
        out = pd.concat([out, prism_row], ignore_index=True)
    out.to_csv(out_csv, index=False)
    return out


def load_prism_panel(prism_path: str, lineage_tag: str = "SKIN"):
    """
    PRISM Repurposing 19Q4 secondary screen (dose-response IC50/AUC).
    Melanoma panel = ccle_name ending in the skin lineage tag (e.g. A375_SKIN).
    """
    prism = pd.read_csv(prism_path, low_memory=False)
    prism = prism[prism["auc"].notna() & prism["ic50"].notna()]
    prism["_lineage"] = prism["ccle_name"].astype(str).str.rsplit("_", n=1).str[-1]
    panel = prism[prism["_lineage"] == lineage_tag][
        ["name", "ccle_name", "ic50", "auc"]
    ].rename(columns={"name": "DRUG_NAME", "ccle_name": "CELL_LINE_NAME",
                      "ic50": "LN_IC50", "auc": "AUC"})
    panel["LN_IC50"] = pd.to_numeric(panel["LN_IC50"], errors="coerce")
    panel["AUC"] = pd.to_numeric(panel["AUC"], errors="coerce")
    panel = panel.dropna()
    # PRISM ic50 is nM linear, GDSC LN_IC50 is natural-log — harmonize to log scale
    panel["LN_IC50"] = np.log(np.maximum(panel["LN_IC50"], 1e-3))
    name_map = {_norm(n): n for n in panel["DRUG_NAME"].unique()}
    return panel, name_map


def run_prism_validation(
    rankings_csv: str,
    prism_csv: str,
    out_csv: str,
    lineage_tag: str = "SKIN",
    top_k: int = 10,
):
    rank_df = pd.read_csv(rankings_csv)
    panel, name_map = load_prism_panel(prism_csv, lineage_tag=lineage_tag)
    res = validate_against_gdsc(rank_df, panel, name_map, top_k=top_k)
    res = {"dataset": f"PRISM_secondary_{lineage_tag}", **res}
    out = pd.DataFrame([res])
    out.to_csv(out_csv, index=False)
    print(f"  [PRISM] mapped={res.get('n_compounds_mapped')} evaluated={res.get('n_compounds_evaluated')} "
          f"rho(IC50)={res.get('spearman_rank_vs_median_IC50')} perm_p={res.get('perm_pval_rank_vs_median_IC50')}")
    return out


if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _needed = {
        "gdsc1_fitted.csv": "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.4/GDSC1_fitted_dose_response_24Jul22.csv",
        "gdsc2_fitted.csv": "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.4/GDSC2_fitted_dose_response_24Jul22.csv",
        "compounds_8.4.csv": "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.4/screened_compounds_rel_8.4.csv",
    }
    _missing = [f for f, url in _needed.items()
                if not os.path.exists(os.path.join(base, "data", "raw_downloads", f))]
    if _missing:
        raise SystemExit(
            "External drug-sensitivity archives not found in data/raw_downloads/:\n  "
            + "\n  ".join(f"{f}  <-  {u}" for f, u in _needed.items() if f in _missing)
            + "\nDownload them from the URLs above (GDSC release 8.4) and retry.")
    run_external_validation(
        rankings_csv=os.path.join(base, "results", "drug_repurposing_rankings.csv"),
        gdsc1_csv=os.path.join(base, "data", "raw_downloads", "gdsc1_fitted.csv"),
        gdsc2_csv=os.path.join(base, "data", "raw_downloads", "gdsc2_fitted.csv"),
        compounds_csv=os.path.join(base, "data", "raw_downloads", "compounds_8.4.csv"),
        out_csv=os.path.join(base, "results", "external_drug_validation_gdsc.csv"),
    )
