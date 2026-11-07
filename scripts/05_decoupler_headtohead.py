#!/usr/bin/env python3
"""
05_decoupler_headtohead.py
==========================
Head-to-head benchmark of the scRegPharm activity-inference step (dropout-aware
analytical ULM) against decoupler's standard ULM under increasing UMI dropout.

Protocol:
  * Binomial UMI-capture downsampling of 2,000 GSE115978 cells, 6 capture
    levels x 3 replicates.
  * Arm A: compute_regulon_activity(..., dropout_aware=True)
    Arm B: decoupler standard ULM
  * Downstream scoring for both arms: bulk activities inferred with the same
    arm, regulon weights = Pearson(bulk activity, resistance phenotype), cell
    score = weighted activity sum; AUROC vs ex-vivo resistance labels.

Run:  python scripts/05_decoupler_headtohead.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from scregpharm.core import compute_regulon_activity  # noqa: E402

DATA = os.path.join(BASE_DIR, "data")
OUT = os.path.join(BASE_DIR, "results", "decoupler_headtohead")
os.makedirs(OUT, exist_ok=True)

CAPTURE_GRID = [
    (1.0000, 0.57),
    (0.0500, 0.71),
    (0.0150, 0.78),
    (0.0040, 0.87),
    (0.0010, 0.94),
    (0.0003, 0.98),
]


def decoupler_ulm(expr_df, net_df):
    """Standard ULM via decoupler (v2.x API: mt.ulm returns (scores, pvals);
    scores = obs x source DataFrame)."""
    import decoupler
    net = net_df.copy()
    if "weight" not in net.columns:
        net["weight"] = 1.0
    mat = expr_df.T  # decoupler: obs x features
    scores, _ = decoupler.mt.ulm(data=mat, net=net, verbose=False)
    # decoupler silently drops all-zero observations at extreme dropout;
    # restore the full sample axis (missing = zero evidence, same rule as ours).
    return scores.reindex(expr_df.columns).fillna(0.0).T  # regulons x cells


def main():
    net_df = pd.read_csv(os.path.join(DATA, "human_regulon_network.csv"))
    sc_expr = pd.read_csv(os.path.join(DATA, "melanoma_sc_expr.csv"), index_col=0)
    sc_raw = pd.read_csv(os.path.join(DATA, "melanoma_sc_raw_counts.csv"), index_col=0)
    sc_meta = pd.read_csv(os.path.join(DATA, "melanoma_sc_meta.csv"), index_col=0)
    bulk_expr = pd.read_csv(os.path.join(DATA, "GSE91061_Riaz_Melanoma_expr.csv"), index_col=0)
    bulk_clin = pd.read_csv(os.path.join(DATA, "GSE91061_Riaz_Melanoma_clinical.csv"), index_col=0)

    sc_raw = sc_raw[~sc_raw.index.duplicated(keep="first")]
    sc_expr.columns = sc_expr.columns.astype(str)
    bulk_expr.columns = bulk_expr.columns.astype(str)
    bulk_clin.index = bulk_clin.index.astype(str)
    common_bulk = [c for c in bulk_expr.columns if c in set(bulk_clin.index)]
    bulk_expr = bulk_expr[common_bulk]
    bulk_clin = bulk_clin.loc[common_bulk]
    # Resistance phenotype from the unified schema (response_bin: 1 = responder).
    # Drop unlabelled patients BEFORE weight estimation — a single NaN would
    # otherwise zero out every regulon weight.
    labelled = bulk_clin["response_bin"].notna().values
    bulk_expr = bulk_expr.loc[:, labelled]
    bulk_clin = bulk_clin.loc[bulk_clin.index[labelled]]
    y_pheno = (1.0 - bulk_clin["response_bin"].astype(float)).values  # 1 = non-responder

    rng = np.random.default_rng(42)
    gt_all = sc_meta["ground_truth_resistant"].values.astype(int)
    sub_idx = rng.choice(len(gt_all), size=2000, replace=False)
    sc_raw_sub = sc_raw.iloc[:, sub_idx].values.astype(np.int64)
    gt_sub = gt_all[sub_idx]
    sub_cols = sc_expr.columns[sub_idx]

    rows = []
    for p_cap, nominal_drop in CAPTURE_GRID:
        for rep in range(3):
            cnt_down = rng.binomial(sc_raw_sub, p_cap if p_cap < 1.0 else 0.92)
            lib = cnt_down.sum(axis=0, keepdims=True) + 1e-8
            X_down = np.log2(cnt_down / lib * 1e4 + 1.0)
            df_down = pd.DataFrame(X_down, index=sc_expr.index, columns=sub_cols)
            zf = float((cnt_down == 0).mean())

            # Arm A: ours (dropout-aware analytical ULM)
            act_a = compute_regulon_activity(df_down, net_df, dropout_aware=True)
            # Arm B: decoupler standard ULM
            try:
                act_b = decoupler_ulm(df_down, net_df)
            except Exception as exc:
                print(f"  decoupler failed @ p_cap={p_cap} rep={rep}: {exc}")
                continue

            # Identical downstream: bulk activities per arm; weights = corr(activity, phenotype)
            aucs = {}
            for tag, act in (("ours", act_a), ("decoupler_ulm", act_b)):
                bulk_act = compute_regulon_activity(bulk_expr, net_df, dropout_aware=False) \
                    if tag == "ours" else decoupler_ulm(bulk_expr, net_df)
                common = act.index.intersection(bulk_act.index)
                bmat = bulk_act.loc[common].values
                w = np.array([sps.pearsonr(bmat[k], y_pheno).statistic
                              for k in range(bmat.shape[0])])
                A = act.loc[common].values.astype(float)
                # NaN-safe scoring (identical rule for both arms): regulons that fail
                # to yield an estimate at high dropout contribute zero evidence.
                w0 = np.where(np.isfinite(w), w, 0.0)
                A0 = np.where(np.isfinite(A), A, 0.0)
                scores = w0 @ A0
                aucs[tag] = roc_auc_score(gt_sub, scores)

            rows.append({"p_cap": p_cap, "obs_zero_frac": round(zf, 3), "rep": rep,
                         "AUROC_ours": round(aucs["ours"], 4),
                         "AUROC_decoupler_ulm": round(aucs["decoupler_ulm"], 4),
                         "delta": round(aucs["ours"] - aucs["decoupler_ulm"], 4)})
            print(f"  p_cap={p_cap:.4f} zf={zf:.3f} rep={rep}: "
                  f"ours={aucs['ours']:.4f} decoupler={aucs['decoupler_ulm']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "decoupler_headtohead.csv"), index=False)
    print("\n=== Head-to-head summary (mean AUROC by capture level) ===")
    s = df.groupby("obs_zero_frac")[["AUROC_ours", "AUROC_decoupler_ulm", "delta"]].mean().round(4)
    print(s.to_string())
    w = sps.wilcoxon(df["AUROC_ours"], df["AUROC_decoupler_ulm"], alternative="greater")
    print(f"\npaired Wilcoxon (ours > decoupler ULM): p={w.pvalue:.4f} over {len(df)} runs")


if __name__ == "__main__":
    main()
