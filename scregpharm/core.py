"""
Core mathematical algorithms for scRegPharm and competing baseline methods:
1. Prior-Guided Dropout-Aware Regulon Manifold Projection (Analytical ULM with Dropout Stabilization)
2. scRegPharm-Map: Graph-Laplacian Regularized Optimal Transport & Woodbury Dual-Ridge Regression
   with Vectorized Permutation FDR Control
3. Baseline Phenotype-Mapping Methods:
   - Scissor (Network-regularized L1+L2 FISTA/coordinate descent across alpha path on raw gene correlation)
   - scPAS (Gene-network sparse regression + permutation Z-score on raw genes)
   - LP_SGL (Label propagation / Sparse group lasso on cell graph)
   - DEGAS / Raw-Correlation Transfer
4. scRegPharm-Spatial: Spatial Niche Projection & Spatial Immune Exclusion Index (SEI)
5. scRegPharm-Drug: Tri-modal Regulon-Guided Pharmacological Reversal & Baselines
   (DrugReSC, ASGARD, Bulk-CMap)
"""

import time
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests


def compute_regulon_activity(
    expr_df: pd.DataFrame,
    net_df: pd.DataFrame,
    min_targets: int = 5,
    scale_across_samples: bool = True,
    dropout_aware: bool = True,
) -> pd.DataFrame:
    """
    Project gene expression matrix (genes x samples/cells) into the Regulon/Pathway
    Activity Manifold using Dropout-Aware Analytical Univariate Linear Model (ULM) t-statistics.

    Mathematical formulation:
    1. Row-wise gene standardization across samples/cells removes baseline expression-level bias.
    2. For sparse single-cell matrices (zero_frac > 0.25) with dropout_aware=True, technical
       zeros are shrinkage-weighted by gene detection probability so extreme dropout does not
       collapse co-regulated target modules.
    3. Column-wise centering & L2-normalization across genes (coupled with column-centered
       prior regulon weights W) computes the exact analytical ULM correlation r_{k,i} and
       student t-statistic t_{k,i} = r_{k,i} * sqrt((G - 2) / (1 - r_{k,i}^2)), removing
       cell-size / total-UMI library bias.
    """
    common_genes = expr_df.index.intersection(net_df["target"].unique())
    if len(common_genes) < 15:
        raise ValueError(f"Too few overlapping genes ({len(common_genes)}) between expression and prior network.")

    sub_expr = expr_df.loc[common_genes]
    sub_net = net_df[net_df["target"].isin(common_genes)].copy()

    counts = sub_net.groupby("source")["target"].count()
    valid_sources = counts[counts >= min_targets].index
    sub_net = sub_net[sub_net["source"].isin(valid_sources)]

    W_df = sub_net.pivot_table(index="target", columns="source", values="weight", aggfunc="mean").fillna(0.0)
    W_df = W_df.reindex(common_genes).fillna(0.0)

    X = sub_expr.values.astype(np.float64)  # (G, N)
    W = W_df.values.astype(np.float64)      # (G, K)
    G, N = X.shape

    # Step 1: Empirical Bayes variance-stabilized gene standardization across cells/samples
    # Prevents sparse high-dropout genes from blowing up column norms while preserving relative activation
    zero_frac = (X == 0.0).mean()
    if N > 3:
        g_mean = X.mean(axis=1, keepdims=True)
        g_var = X.var(axis=1, keepdims=True)
        if dropout_aware and zero_frac > 0.25:
            med_var = np.median(g_var[g_var > 0]) if np.any(g_var > 0) else 1.0
            eb_prior = 0.5 + 1.5 * (max(zero_frac - 0.50, 0.0) / 0.50)
            X_row = (X - g_mean) / np.sqrt(g_var + eb_prior * med_var + 1e-8)
        else:
            X_row = (X - g_mean) / (np.sqrt(g_var) + 1e-8)
    else:
        X_row = X

    # Step 2: Exact Analytical ULM (column-wise centering & L2 normalization across G genes)
    X_c = X_row - X_row.mean(axis=0, keepdims=True)
    X_c_norm = X_c / (np.linalg.norm(X_c, axis=0, keepdims=True) + 1e-8)

    W_c = W - W.mean(axis=0, keepdims=True)
    W_c_norm = W_c / (np.linalg.norm(W_c, axis=0, keepdims=True) + 1e-8)

    # Pearson correlation r_{k, i} in [-1, 1] of shape (K, N)
    r_mat = np.clip(W_c_norm.T @ X_c_norm, -0.9999, 0.9999)
    dof = max(G - 2, 1)
    t_stat = r_mat * np.sqrt(dof / np.maximum(1.0 - r_mat ** 2, 1e-12))

    if scale_across_samples and N > 3:
        t_mean = t_stat.mean(axis=1, keepdims=True)
        t_std = t_stat.std(axis=1, keepdims=True) + 1e-8
        act_mat = (t_stat - t_mean) / t_std
    else:
        act_mat = t_stat

    return pd.DataFrame(act_mat, index=W_df.columns, columns=expr_df.columns)


def _build_Normalized_Laplacian(features: np.ndarray, n_neighbors: int = 15):
    """
    Construct symmetric normalized Graph Laplacian L = I - D^{-1/2} A D^{-1/2}
    from feature matrix (N_cells x D_features).
    Uses L2-normalized Euclidean NearestNeighbors (mathematically identical to cosine
    distance ||u - v||^2 = 2(1 - cos(u,v)), avoiding dense N x N pairwise allocation).
    """
    N = features.shape[0]
    k = min(n_neighbors, max(N - 1, 1))
    feat_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    algo = "kd_tree" if feat_norm.shape[1] <= 18 else "auto"
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm=algo)
    nn.fit(feat_norm)
    distances, indices = nn.kneighbors(feat_norm)

    rows = np.repeat(np.arange(N), k)
    cols = indices[:, 1:].reshape(-1)
    dists = distances[:, 1:].reshape(-1)
    sigma = np.median(dists) + 1e-6
    weights = np.exp(-(dists ** 2) / (2.0 * sigma ** 2))

    A = sparse.csr_matrix((weights, (rows, cols)), shape=(N, N))
    A = 0.5 * (A + A.T)

    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(np.maximum(deg, 1e-8), -0.5)
    D_inv_sqrt = sparse.diags(deg_inv_sqrt)
    I = sparse.eye(N, format="csr")
    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    return L, A, D_inv_sqrt


def _sinkhorn_coupling(cost_matrix: np.ndarray, reg: float = 0.08, n_iters: int = 25) -> np.ndarray:
    """
    Entropic Optimal Transport (Sinkhorn-Knopp algorithm) between N cells and M bulk samples.
    """
    N, M = cost_matrix.shape
    C = cost_matrix - cost_matrix.min()
    C_std = C.std() + 1e-8
    C = C / C_std

    K = np.exp(-C / max(reg, 0.01))
    K = np.maximum(K, 1e-12)

    a = np.ones(N, dtype=np.float64) / float(N)
    b = np.ones(M, dtype=np.float64) / float(M)
    u = np.ones(N, dtype=np.float64) / float(N)
    v = np.ones(M, dtype=np.float64) / float(M)

    for _ in range(n_iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)

    P = (u[:, None] * K) * v[None, :]
    return P * float(N)


def scregpharm_map(
    sc_expr: pd.DataFrame,
    bulk_expr: pd.DataFrame,
    phenotype: np.ndarray,
    net_df: pd.DataFrame,
    n_neighbors: int = 20,
    ot_weight: float = 0.15,
    ridge_lambda: float = 0.25,
    diffusion_alpha: float = 0.35,
    diffusion_steps: int = 5,
    n_perms: int = 200,
    fdr_threshold: float = 0.05,
    use_regulon: bool = True,
    use_ot: bool = True,
    use_laplacian: bool = True,
    use_dual_ridge: bool = True,
    return_regulon_df: bool = True,
    seed: int = 42,
) -> dict:
    """
    scRegPharm Core Algorithm:
    Projects single-cell and bulk transcriptomes onto the Prior Regulon Manifold via
    Analytical ULM and Zero-Copy Dual-Bulk Prior-Regulon Subspace Denoising, computes
    Sinkhorn Optimal Transport + Cosine Alignment coupling, solves trace-scaled Dual-Ridge
    regression in M x M sample space via the Woodbury identity, applies multi-step
    Graph-Laplacian spectral diffusion on the low-dimensional Regulon-Bulk Manifold,
    and calibrates empirical FDR via vectorized label permutations.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    common_genes = sc_expr.index.intersection(bulk_expr.index)
    X_sc = sc_expr.to_numpy(copy=False) if len(common_genes) == len(sc_expr.index) else sc_expr.loc[common_genes].to_numpy(copy=False)
    X_b = bulk_expr.to_numpy(copy=False) if len(common_genes) == len(bulk_expr.index) else bulk_expr.loc[common_genes].to_numpy(copy=False)
    G, N = X_sc.shape
    M = X_b.shape[1]

    # Sample-wise sparsity estimate (fast slice for large N)
    step_z = max(N // 500, 1)
    z_frac = float((X_sc[:, ::step_z] == 0.0).mean())

    y = np.asarray(phenotype, dtype=np.float64).flatten()
    y_z = (y - y.mean()) / (y.std() + 1e-8)

    # Zero-copy column norms via einsum variance identity: sum((x - mu)^2) = sum(x^2) - (sum(x)^2)/G
    sc_sum = np.einsum("gn->n", X_sc)
    sc_sq = np.einsum("gn,gn->n", X_sc, X_sc)
    sc_norm = np.sqrt(np.maximum(sc_sq - (sc_sum ** 2) / float(G), 1e-8))

    b_mean = X_b.mean(axis=0, keepdims=True)
    X_b_c = X_b - b_mean
    b_norm = np.linalg.norm(X_b_c, axis=0, keepdims=True) + 1e-8
    X_b_z = X_b_c / b_norm

    sub_net = net_df[net_df["target"].isin(common_genes)]
    W_df = sub_net.pivot_table(index="target", columns="source", values="weight", aggfunc="mean").fillna(0.0).reindex(common_genes).fillna(0.0)
    W = W_df.values.astype(np.float64)
    W_c = W - W.mean(axis=0, keepdims=True)
    W_cn = np.ascontiguousarray(W_c / (np.linalg.norm(W_c, axis=0, keepdims=True) + 1e-8))

    # Zero-copy Analytical ULM projection into K x N regulon space (since W_cn is column-centered, W_cn^T @ 1 = 0)
    Z_sc_reg = (np.ascontiguousarray(W_cn.T) @ X_sc) / sc_norm[None, :]  # (K, N)
    Z_b_reg = np.ascontiguousarray(W_cn.T) @ X_b_z                       # (K, M)

    if use_regulon:
        r_reg = (Z_b_reg @ y_z) / (Z_b_reg.std(axis=1) * float(M) + 1e-8)
        w_reg_rel = np.abs(r_reg) ** 1.5
        w_reg_rel = w_reg_rel / (w_reg_rel.sum() + 1e-8)

        den_gamma = 0.8 + 4.5 * (max(z_frac - 0.55, 0.0) / 0.45) ** 1.5
        # Apply symmetric operator (I + den_gamma * W_cn D_rel W_cn^T) to G x M bulk matrix
        U_b = X_b_z + (2.0 * den_gamma) * (W_cn @ (w_reg_rel[:, None] * Z_b_reg))
        U_b_c = U_b - U_b.mean(axis=0, keepdims=True)
        U_b_z = U_b_c / (np.linalg.norm(U_b_c, axis=0, keepdims=True) + 1e-8)

        # Multiply (M x G) @ (G x N) and transpose to avoid copying G x N X_sc
        S_corr = np.ascontiguousarray((np.ascontiguousarray(U_b_z.T) @ X_sc).T) / sc_norm[:, None]
        F_raw = np.hstack([
            0.35 * (Z_sc_reg.T / (Z_sc_reg.std() + 1e-8)),
            S_corr / (S_corr.std() + 1e-8),
        ])
    else:
        # Ablation w/o Prior Regulon Projection: standard row-standardized raw gene space
        X_b_row = (X_b - X_b.mean(axis=1, keepdims=True)) / (X_b.std(axis=1, keepdims=True) + 1e-8) / np.sqrt(float(G))
        X_b_row_c = X_b_row - X_b_row.mean(axis=0, keepdims=True)
        S_corr = np.ascontiguousarray((np.ascontiguousarray(X_b_row_c.T) @ X_sc).T) / sc_norm[:, None]
        F_raw = S_corr / (S_corr.std() + 1e-8)

    # Project manifold features to top 12 eigendirections for KD-tree neighbor search
    n_comp = min(12, F_raw.shape[1])
    C_f = (F_raw.T @ F_raw) / float(N)
    _, eigvecs = np.linalg.eigh(C_f)
    F_manifold = np.ascontiguousarray(F_raw @ eigvecs[:, -n_comp:])

    if use_ot and ot_weight > 0:
        cost_mat = 1.0 - S_corr
        P_ot = _sinkhorn_coupling(cost_mat, reg=0.25, n_iters=20)
        P_ot_s = (P_ot - P_ot.mean()) / (P_ot.std() + 1e-8) * (S_corr.std() + 1e-8)
        S_hybrid = (1.0 - ot_weight) * S_corr + ot_weight * P_ot_s
    else:
        S_hybrid = S_corr

    G_mat = (S_hybrid.T @ S_hybrid) / float(N)
    tr_val = float(np.trace(G_mat)) + 1e-8
    if use_dual_ridge:
        gram_M = G_mat + (ridge_lambda * tr_val) * np.eye(M, dtype=np.float64)
        dual_weights = np.linalg.solve(gram_M, y_z)
        beta_init = (S_hybrid @ dual_weights) / np.sqrt(float(M))
    else:
        gram_M = G_mat + (0.0025 * tr_val) * np.eye(M, dtype=np.float64)
        dual_weights = np.linalg.solve(gram_M, y_z)
        beta_init = (S_hybrid @ dual_weights) / np.sqrt(float(M))

    eff_alpha = diffusion_alpha + 0.35 * (max(z_frac - 0.55, 0.0) / 0.45) ** 1.2
    eff_alpha = min(eff_alpha, 0.72)
    if use_laplacian and eff_alpha > 0 and N > n_neighbors:
        _, A_adj, D_inv_sqrt = _build_Normalized_Laplacian(F_manifold, n_neighbors=n_neighbors)
        P_norm = D_inv_sqrt @ A_adj @ D_inv_sqrt
        beta_smooth = beta_init.copy()
        for _ in range(diffusion_steps):
            beta_smooth = (1.0 - eff_alpha) * beta_init + eff_alpha * (P_norm @ beta_smooth)
    else:
        beta_smooth = beta_init.copy()
        P_norm = None

    Y_perm = np.zeros((M, n_perms), dtype=np.float64)
    for p in range(n_perms):
        Y_perm[:, p] = rng.permutation(y_z)

    Dual_perm = np.linalg.solve(gram_M, Y_perm)
    Beta_perm_init = (S_hybrid @ Dual_perm) / np.sqrt(float(M))

    Beta_perm = Beta_perm_init.copy()
    if P_norm is not None:
        for _ in range(diffusion_steps):
            Beta_perm = (1.0 - eff_alpha) * Beta_perm_init + eff_alpha * (P_norm @ Beta_perm)

    null_mean = Beta_perm.mean(axis=1)
    perm_var = Beta_perm.var(axis=1)
    null_std = np.sqrt(0.15 * perm_var + 0.85 * float(np.median(perm_var)) + 1e-8)
    pas_z = (beta_smooth - null_mean) / null_std

    p_vals = 2.0 * stats.norm.sf(np.abs(pas_z))
    _, fdr_vals, _, _ = multipletests(p_vals, alpha=fdr_threshold, method="fdr_bh")

    labels = np.zeros(N, dtype=int)
    labels[(pas_z > 0.75) & (fdr_vals < max(fdr_threshold, 0.05))] = 1
    labels[(pas_z < -0.75) & (fdr_vals < max(fdr_threshold, 0.05))] = -1

    if np.sum(labels == 1) < 20:
        labels[pas_z > 0.85] = 1
    if np.sum(labels == -1) < 20:
        labels[pas_z < -0.85] = -1

    if return_regulon_df:
        sc_reg = compute_regulon_activity(sc_expr, net_df, dropout_aware=True)
        bulk_reg = compute_regulon_activity(bulk_expr, net_df, dropout_aware=False)
    else:
        sc_reg = None
        bulk_reg = None

    runtime = time.perf_counter() - t0
    return {
        "pas_score": pas_z,
        "beta_raw": beta_smooth,
        "p_values": p_vals,
        "fdr": fdr_vals,
        "cell_labels": labels,
        "sc_regulon": sc_reg,
        "bulk_regulon": bulk_reg,
        "runtime_sec": runtime,
    }


def run_scissor_baseline(
    sc_expr: pd.DataFrame,
    bulk_expr: pd.DataFrame,
    phenotype: np.ndarray,
    alpha_grid: tuple = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2),
    lambda_net: float = 0.15,
    n_neighbors: int = 15,
    max_iter: int = 55,
) -> dict:
    """
    Baseline 1: Scissor (Sun et al., Nature Biotechnology 2022).
    Constructs the raw-gene correlation matrix S (N x M) and raw-gene SNN/KNN graph Laplacian L (N x N),
    then solves the network-regularized L1 + L2 objective across the alpha path via proximal
    gradient descent (FISTA/ISTA) until the selected cell fraction criteria are satisfied.
    """
    t0 = time.perf_counter()
    common_genes = sc_expr.index.intersection(bulk_expr.index)
    X_sc = sc_expr.loc[common_genes].values.T.astype(np.float64)    # (N, G)
    X_bulk = bulk_expr.loc[common_genes].values.T.astype(np.float64)  # (M, G)
    N, G = X_sc.shape
    M = X_bulk.shape[0]

    X_sc_z = (X_sc - X_sc.mean(axis=1, keepdims=True)) / (X_sc.std(axis=1, keepdims=True) + 1e-8)
    X_bulk_z = (X_bulk - X_bulk.mean(axis=1, keepdims=True)) / (X_bulk.std(axis=1, keepdims=True) + 1e-8)

    S_raw = (X_sc_z @ X_bulk_z.T) / float(G)

    L_norm, A_knn, _ = _build_Normalized_Laplacian(X_sc_z, n_neighbors=n_neighbors)
    # Construct shared nearest neighbor (SNN) overlap Laplacian as in Seurat/Scissor
    A_snn = (A_knn @ A_knn.T).tocsr()
    deg_snn = np.array(A_snn.sum(axis=1)).flatten()
    D_snn_inv = sparse.diags(np.power(np.maximum(deg_snn, 1e-8), -0.5))
    L_snn = sparse.eye(N, format="csr") - D_snn_inv @ A_snn @ D_snn_inv
    L_comb = 0.5 * (L_norm + L_snn)

    y = np.asarray(phenotype, dtype=np.float64).flatten()
    y_z = (y - y.mean()) / (y.std() + 1e-8)

    step_size = 0.015
    beta_best = np.zeros(N, dtype=np.float64)
    for alpha in alpha_grid:
        beta = (S_raw @ y_z) / float(M)
        l1_thresh = 0.12 * step_size * lambda_net * alpha
        l2_weight = lambda_net * (1.0 - alpha)
        for _ in range(max_iter):
            residual = (S_raw.T @ beta) - y_z
            grad_loss = (S_raw @ residual) / float(M)
            grad_net = l2_weight * np.asarray(L_comb @ beta).flatten()
            v = beta - step_size * (grad_loss + grad_net)
            beta = np.sign(v) * np.maximum(np.abs(v) - l1_thresh, 0.0)
        beta_best = beta
        if 0.05 <= np.mean(np.abs(beta) > 1e-4) <= 0.35:
            break

    beta_z = (beta_best - beta_best.mean()) / (beta_best.std() + 1e-8)
    labels = np.zeros(N, dtype=int)
    labels[beta_z > 0.90] = 1
    labels[beta_z < -0.90] = -1

    runtime = time.perf_counter() - t0
    return {
        "pas_score": beta_z,
        "cell_labels": labels,
        "runtime_sec": runtime,
    }


def run_scpas_baseline(
    sc_expr: pd.DataFrame,
    bulk_expr: pd.DataFrame,
    phenotype: np.ndarray,
    n_neighbors: int = 15,
    n_perms: int = 200,
    seed: int = 42,
) -> dict:
    """
    Baseline 2: scPAS (Xie et al., Briefings in Bioinformatics 2025).
    Operates in raw gene space with gene-gene network regularization and permutation Z-test.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    common_genes = sc_expr.index.intersection(bulk_expr.index)
    X_sc = sc_expr.loc[common_genes].values.astype(np.float64)      # (G, N)
    X_bulk = bulk_expr.loc[common_genes].values.astype(np.float64)  # (G, M)
    G, N = X_sc.shape
    M = X_bulk.shape[1]

    X_bulk_z = (X_bulk - X_bulk.mean(axis=1, keepdims=True)) / (X_bulk.std(axis=1, keepdims=True) + 1e-8)
    X_sc_z = (X_sc - X_sc.mean(axis=1, keepdims=True)) / (X_sc.std(axis=1, keepdims=True) + 1e-8)

    y = np.asarray(phenotype, dtype=np.float64).flatten()
    y_z = (y - y.mean()) / (y.std() + 1e-8)

    # Gene-level network-regularized weights in raw gene space (Xie et al., BiB 2025)
    # Build G x G gene-gene kNN similarity network from scRNA-seq matrix X_sc_z (G x N)
    L_g, A_g, D_g = _build_Normalized_Laplacian(X_sc_z, n_neighbors=min(15, G - 1))
    P_g = D_g @ A_g @ D_g
    w_gene = (X_bulk_z @ y_z) / float(M)
    for _ in range(18):
        grad_b = (X_bulk_z @ ((X_bulk_z.T @ w_gene) / float(G) - y_z)) / float(M)
        grad_g = 0.20 * np.asarray(L_g @ w_gene).flatten()
        v_g = w_gene - 0.08 * (grad_b + grad_g)
        w_gene = np.sign(v_g) * np.maximum(np.abs(v_g) - 0.002, 0.0)
    w_gene_smooth = 0.75 * w_gene + 0.25 * (P_g @ w_gene)

    raw_pas = (X_sc_z.T @ w_gene_smooth) / np.sqrt(float(G))

    # Permutation null in raw gene space (random gene permutations as in scPAS)
    Perm_W = np.zeros((G, n_perms), dtype=np.float64)
    for p in range(n_perms):
        Perm_W[:, p] = rng.permutation(w_gene_smooth)
    Perm_PAS = (X_sc_z.T @ Perm_W) / np.sqrt(float(G))

    null_mu = Perm_PAS.mean(axis=1)
    null_sd = Perm_PAS.std(axis=1) + 1e-8
    pas_z = (raw_pas - null_mu) / null_sd

    p_vals = 2.0 * stats.norm.sf(np.abs(pas_z))
    _, fdr_vals, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")

    labels = np.zeros(N, dtype=int)
    labels[(pas_z > 0) & (fdr_vals < 0.05)] = 1
    labels[(pas_z < 0) & (fdr_vals < 0.05)] = -1
    if np.sum(labels == 1) < 15:
        labels[pas_z > 1.96] = 1
    if np.sum(labels == -1) < 15:
        labels[pas_z < -1.96] = -1

    runtime = time.perf_counter() - t0
    return {
        "pas_score": pas_z,
        "cell_labels": labels,
        "runtime_sec": runtime,
    }


def run_degas_corr_baseline(
    sc_expr: pd.DataFrame,
    bulk_expr: pd.DataFrame,
    phenotype: np.ndarray,
) -> dict:
    """
    Baseline 3: Direct raw-gene correlation phenotype transfer (DEGAS-like linear projection).
    """
    t0 = time.perf_counter()
    common_genes = sc_expr.index.intersection(bulk_expr.index)
    X_sc = sc_expr.loc[common_genes].values.T.astype(np.float64)
    X_bulk = bulk_expr.loc[common_genes].values.T.astype(np.float64)
    N, G = X_sc.shape

    X_sc_z = (X_sc - X_sc.mean(axis=1, keepdims=True)) / (X_sc.std(axis=1, keepdims=True) + 1e-8)
    X_bulk_z = (X_bulk - X_bulk.mean(axis=1, keepdims=True)) / (X_bulk.std(axis=1, keepdims=True) + 1e-8)

    S = (X_sc_z @ X_bulk_z.T) / float(G)
    y = np.asarray(phenotype, dtype=np.float64).flatten()
    y_z = (y - y.mean()) / (y.std() + 1e-8)

    score = S @ y_z
    score_z = (score - score.mean()) / (score.std() + 1e-8)
    labels = np.zeros(N, dtype=int)
    labels[score_z > 0.90] = 1
    labels[score_z < -0.90] = -1

    return {
        "pas_score": score_z,
        "cell_labels": labels,
        "runtime_sec": time.perf_counter() - t0,
    }


def run_lp_sgl_baseline(
    sc_expr: pd.DataFrame,
    bulk_expr: pd.DataFrame,
    phenotype: np.ndarray,
    n_neighbors: int = 15,
) -> dict:
    """
    Baseline 4: Label Propagation / Sparse Group Lasso (LP_SGL) on raw-gene KNN graph.
    """
    t0 = time.perf_counter()
    common_genes = sc_expr.index.intersection(bulk_expr.index)
    X_sc = sc_expr.loc[common_genes].values.T.astype(np.float64)
    X_bulk = bulk_expr.loc[common_genes].values.T.astype(np.float64)
    N, G = X_sc.shape

    X_sc_z = (X_sc - X_sc.mean(axis=1, keepdims=True)) / (X_sc.std(axis=1, keepdims=True) + 1e-8)
    X_bulk_z = (X_bulk - X_bulk.mean(axis=1, keepdims=True)) / (X_bulk.std(axis=1, keepdims=True) + 1e-8)

    S = (X_sc_z @ X_bulk_z.T) / float(G)
    y = np.asarray(phenotype, dtype=np.float64).flatten()
    y_z = (y - y.mean()) / (y.std() + 1e-8)
    init_score = S @ y_z

    _, A_adj, D_inv_sqrt = _build_Normalized_Laplacian(X_sc_z, n_neighbors=n_neighbors)
    P_norm = D_inv_sqrt @ A_adj @ D_inv_sqrt
    prop = init_score.copy()
    for _ in range(5):
        prop = 0.5 * init_score + 0.5 * (P_norm @ prop)

    prop_z = (prop - prop.mean()) / (prop.std() + 1e-8)
    labels = np.zeros(N, dtype=int)
    labels[prop_z > 0.90] = 1
    labels[prop_z < -0.90] = -1

    return {
        "pas_score": prop_z,
        "cell_labels": labels,
        "runtime_sec": time.perf_counter() - t0,
    }


def scregpharm_spatial_score(
    visium_expr: pd.DataFrame,
    visium_coords: pd.DataFrame,
    sc_regulon_sig: pd.Series,
    net_df: pd.DataFrame,
    spatial_k: int = 6,
    smooth_alpha: float = 0.35,
) -> dict:
    """
    scRegPharm-Spatial:
    Projects 10x Visium spatial transcriptomics onto the Regulon Manifold, applies
    spatial graph smoothing across immediate hex-grid neighbors, and computes the
    Spatial Phenotype Association Score (sPAS) and Niche Regulon Programs.
    """
    vis_reg = compute_regulon_activity(visium_expr, net_df, dropout_aware=False)

    xy = visium_coords[["x", "y"]].values.astype(np.float64)
    n_spots = xy.shape[0]
    nn = NearestNeighbors(n_neighbors=min(spatial_k + 1, n_spots), metric="euclidean")
    nn.fit(xy)
    dists, idx = nn.kneighbors(xy)

    rows = np.repeat(np.arange(n_spots), spatial_k)
    cols = idx[:, 1:].reshape(-1)
    d_vals = dists[:, 1:].reshape(-1)
    sigma = np.median(d_vals) + 1e-6
    w_vals = np.exp(-(d_vals ** 2) / (2.0 * sigma ** 2))

    A_sp = sparse.csr_matrix((w_vals, (rows, cols)), shape=(n_spots, n_spots))
    A_sp = 0.5 * (A_sp + A_sp.T)
    deg = np.array(A_sp.sum(axis=1)).flatten()
    D_inv = sparse.diags(1.0 / np.maximum(deg, 1e-8))
    P_sp = D_inv @ A_sp

    R_mat = vis_reg.values.T  # (S, K)
    R_smooth = (1.0 - smooth_alpha) * R_mat + smooth_alpha * (P_sp @ R_mat)
    vis_reg_smooth = pd.DataFrame(R_smooth.T, index=vis_reg.index, columns=vis_reg.columns)

    common_reg = vis_reg_smooth.index.intersection(sc_regulon_sig.index)
    w_reg = sc_regulon_sig.loc[common_reg].values
    w_reg = w_reg / (np.linalg.norm(w_reg) + 1e-8)

    spas_raw = vis_reg_smooth.loc[common_reg].values.T @ w_reg
    spas_z = (spas_raw - spas_raw.mean()) / (spas_raw.std() + 1e-8)

    return {
        "sPAS": spas_z,
        "visium_regulon": vis_reg_smooth,
        "spatial_graph": P_sp,
    }


def _weighted_ks_reversal(disease_sig: pd.Series, drug_profile: pd.Series, top_n: int = 80) -> float:
    """
    Weighted Kolmogorov-Smirnov / CMap-style enrichment score measuring how strongly
    a drug perturbation downregulates the top upregulated disease genes and upregulates
    the top downregulated disease genes.
    """
    common = disease_sig.index.intersection(drug_profile.index)
    d_sig = disease_sig.loc[common].sort_values(ascending=False)
    up_genes = d_sig.head(top_n).index
    dn_genes = d_sig.tail(top_n).index

    drug_s = drug_profile.loc[common]
    z_up = drug_s.loc[up_genes].mean()
    z_dn = drug_s.loc[dn_genes].mean()
    return float(z_dn - z_up)


def scregpharm_drug_prioritization(
    sc_regulon_diff: pd.Series,
    sc_gene_diff: pd.Series,
    lincs_drug_mat: pd.DataFrame,
    net_df: pd.DataFrame,
    crispr_dep_scores: pd.Series = None,
    drug_targets_map: dict = None,
    w_regulon: float = 0.35,
    w_gene: float = 0.35,
    w_crispr: float = 0.30,
) -> pd.DataFrame:
    """
    scRegPharm-Drug Tri-Modal Pharmacological Reversal Engine:
    1. Regulon-Level Reversal (S_reg): Projects LINCS L1000 chemical perturbation signatures
       onto the Regulon Manifold and computes weighted anti-cosine reversal against the
       scRegPharm+ resistance regulon signature.
    2. Gene-Level Weighted KS Reversal (S_gene): CMap-style reversal of top differential genes.
    3. DepMap CRISPR Target Essentiality Coupling (S_crispr): Evaluates whether the drug's
       direct molecular targets are genetically essential (negative Chronos score) in
       melanoma cell lines and upregulated in scRegPharm+ resistant cells.
    """
    drug_reg_mat = compute_regulon_activity(lincs_drug_mat, net_df, scale_across_samples=False, dropout_aware=False)

    common_regs = sc_regulon_diff.index.intersection(drug_reg_mat.index)
    disease_reg_vec = sc_regulon_diff.loc[common_regs].values
    weights_reg = np.abs(disease_reg_vec) / (np.sum(np.abs(disease_reg_vec)) + 1e-8)

    drug_names = lincs_drug_mat.columns.tolist()
    s_reg_list = []
    s_gene_list = []
    s_crispr_list = []

    disease_reg_s = sc_regulon_diff.copy()
    gene_diff_s = (sc_gene_diff - sc_gene_diff.mean()) / (sc_gene_diff.std() + 1e-8)

    for d_name in drug_names:
        d_reg_vec = drug_reg_mat.loc[common_regs, d_name].values
        num = -np.sum(weights_reg * disease_reg_vec * d_reg_vec)
        denom = np.sqrt(np.sum(weights_reg * (disease_reg_vec ** 2))) * (np.sqrt(np.sum(weights_reg * (d_reg_vec ** 2))) + 0.85) + 1e-8
        s_reg_list.append(float(num / denom))

        s_gene_list.append(_weighted_ks_reversal(gene_diff_s, lincs_drug_mat[d_name], top_n=60))

        c_score = 0.0
        if crispr_dep_scores is not None and drug_targets_map is not None:
            targets = drug_targets_map.get(d_name, [])
            t_scores = []
            for t in targets:
                ess = -float(crispr_dep_scores.get(t, 0.0))
                up_reg = float(disease_reg_s.get(t, 0.0))
                up_gene = float(gene_diff_s.get(t, 0.0))
                up_rel = max(up_reg, up_gene, 0.25)
                if ess > 0:
                    t_scores.append(ess * max(up_rel, 0.25))
            if t_scores:
                c_score = float(np.mean(t_scores))
        s_crispr_list.append(c_score)

    df = pd.DataFrame({
        "drug_name": drug_names,
        "regulon_reversal_raw": s_reg_list,
        "gene_wks_raw": s_gene_list,
        "crispr_synergy_raw": s_crispr_list,
    })

    for col in ["regulon_reversal_raw", "gene_wks_raw", "crispr_synergy_raw"]:
        z_col = col.replace("_raw", "_z")
        std_val = df[col].std()
        if std_val > 1e-8:
            df[z_col] = (df[col] - df[col].mean()) / std_val
        else:
            df[z_col] = 0.0

    df["scRegPharm_Drug_Score"] = (
        w_regulon * df["regulon_reversal_z"]
        + w_gene * df["gene_wks_z"]
        + w_crispr * df["crispr_synergy_z"]
    )
    df["p_value"] = stats.norm.sf(df["scRegPharm_Drug_Score"])
    _, df["fdr"], _, _ = multipletests(df["p_value"], alpha=0.05, method="fdr_bh")
    df = df.sort_values("scRegPharm_Drug_Score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df, drug_reg_mat


def run_drugresc_baseline(sc_gene_diff: pd.Series, lincs_drug_mat: pd.DataFrame) -> pd.Series:
    """
    Baseline Drug 1: DrugReSC (Briefings in Bioinformatics 2024) — Pareto / rank-based
    gene-level reversal on single-cell logFC signature without Regulon projection.
    """
    common = sc_gene_diff.index.intersection(lincs_drug_mat.index)
    sig = sc_gene_diff.loc[common]
    scores = {}
    for d in lincs_drug_mat.columns:
        scores[d] = _weighted_ks_reversal(sig, lincs_drug_mat.loc[common, d], top_n=100)
    s = pd.Series(scores)
    return (s - s.mean()) / (s.std() + 1e-8)


def run_asgard_baseline(sc_gene_diff_by_cluster: dict, lincs_drug_mat: pd.DataFrame) -> pd.Series:
    """
    Baseline Drug 2: ASGARD — Multi-cluster gene-level CMap aggregation across resistant clusters.
    """
    cluster_scores = []
    for _, sig in sc_gene_diff_by_cluster.items():
        s = run_drugresc_baseline(sig, lincs_drug_mat)
        cluster_scores.append(s)
    combined = pd.concat(cluster_scores, axis=1).mean(axis=1)
    return (combined - combined.mean()) / (combined.std() + 1e-8)


def run_bulk_cmap_baseline(bulk_gene_diff: pd.Series, lincs_drug_mat: pd.DataFrame) -> pd.Series:
    """
    Baseline Drug 3: Traditional Bulk-CMap (reversing bulk differential expression directly).
    """
    return run_drugresc_baseline(bulk_gene_diff, lincs_drug_mat)
