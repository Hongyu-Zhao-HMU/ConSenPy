import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import stats
import scipy.sparse as sp
from statsmodels.stats.multitest import multipletests
from .utils import VAE_Trainer, compute_bulk_similarity_score, logistic_regression_scoring, _compute_core_sgs


def preprocess_data(adata_sc):
    """
        Preparing model training data ...
    """
    sc.pp.filter_genes(adata_sc, min_cells=max(10, int(adata_sc.n_obs * 0.001)))
    if "counts" not in adata_sc.layers:
        adata_sc.layers["counts"] = adata_sc.X.copy()
    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    senescence_genes_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "senescence_genes.csv"
    )
    sen_genes = set(pd.read_csv(senescence_genes_path)["Gene"].astype(str))
    common_genes = sorted(sen_genes.intersection(adata_sc.var_names))
    adata_sc = adata_sc[:, common_genes].copy()
    return adata_sc


def calculate_cell_senescence_scores(adata_sc, batch_key=None, epochs=500, z_dim=32, layer_dims=None,
                                     lr=1e-3, dropout=0.2, seed=12, patience=20):
    """
       Model training and cellular aging score prediction
    """
    print(f"Starting model training ...")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    is_spatial = "spatial" in adata_sc.obsm.keys()
    if is_spatial:
        print("Detected 'spatial' in adata_sc.obsm. Switching to Spatial GCN Model.")
    else:
        print("No 'spatial' found in adata_sc.obsm. Using standard Single-Cell MLP Model.")
    if layer_dims is None:
        layer_dims = [256, 128]
    config = {
        "vae_params": {
            "epochs": epochs,
            "z_dim": z_dim,
            "layer_dims": layer_dims,
            "lr": lr,
            "dropout": dropout,
            "device": DEVICE,
            "seed": seed,
            "patience": patience,
            "is_spatial": is_spatial
        }
    }
    vae = VAE_Trainer(**config['vae_params'])
    vae.initialize(adata_sc, batch_key=batch_key)
    vae.train()
    print("Computing cellular aging score ...")
    Z = vae.get_latent()
    adata_sc.obsm["X_vae"] = Z
    bulk_path = os.path.join(os.path.dirname(__file__),"data","bulk_reference.csv")
    df = pd.read_csv(bulk_path,index_col=0)
    expression_matrix = df.drop("label",axis=1).values
    labels = df["label"].values
    adata_bulk = ad.AnnData(
        X=expression_matrix,
        obs=pd.DataFrame(
            {
                "label": labels
            },
            index=df.index
        ),
        var=pd.DataFrame(
            index=df.columns[:-1]
        )
    )
    bulk_labels = adata_bulk.obs["label"].astype(int).values
    s_score = compute_bulk_similarity_score(adata_sc, adata_bulk, bulk_labels)
    s_score = np.asarray(s_score, dtype=np.float64)
    adata_sc.obs["s_bulk_score"] = s_score
    final_score = logistic_regression_scoring(Z, s_score)
    if "CDKN1A" in adata_sc.var_names:
        anchor = adata_sc[:, "CDKN1A"].X
        anchor = anchor.toarray().ravel() if sp.issparse(anchor) else np.asarray(anchor).ravel()
        anchor = anchor.astype(float)
        mask = np.isfinite(final_score) & np.isfinite(anchor)
        if anchor.size > 0 and np.nanstd(anchor) > 0 and mask.sum() > 1:
            rho, _ = stats.spearmanr(final_score[mask], anchor[mask])
            if np.isfinite(rho) and rho < 0:
                final_score = -final_score
    final_score = (final_score - np.min(final_score)) / (np.max(final_score) - np.min(final_score) + 1e-8)
    adata_sc.obs["SenseScore"] = final_score
    adata_sc.obs.drop(columns=[None], errors="ignore", inplace=True)
    return adata_sc



def calculate_differential_sgs(adata_sc, groupby="Condition", label_1="Cond.1", label_2="Cond.2",
                               score_col="SenseScore", level="cell", sample_col="Sample",
                               min_cells_per_sample=500, min_samples_for_sample=20,
                               n_perms=500, min_delta_abs=1, min_sgs_threshold=1, condition_filter="both",
                               n_bins=5):
    np.random.seed(42)
    print(f"--- Differential SGS Analysis: {label_1} vs {label_2} | Level: {level.upper()} | Bins: {n_bins} ---")
    var_idx_all = adata_sc.var_names
    if level == "cell":
        mask_comb = (adata_sc.obs[groupby] == label_1) | (adata_sc.obs[groupby] == label_2)
        adata_comb = adata_sc[mask_comb].copy()
        adata_comb.obs['cell_senescence_bin'] = -1
        for condition in [label_1, label_2]:
            cond_mask = adata_comb.obs[groupby] == condition
            cond_scores = adata_comb.obs.loc[cond_mask, score_col].values
            try:
                cond_bins = pd.qcut(cond_scores, q=n_bins, labels=False)
            except ValueError:
                cond_bins = pd.qcut(pd.Series(cond_scores).rank(level='first'), q=n_bins, labels=False).values
            adata_comb.obs.loc[cond_mask, 'cell_senescence_bin'] = cond_bins
        actual_labels = adata_comb.obs[groupby].values
        X_comb = adata_comb.X
        X_comb_bin = (X_comb > 0).astype(np.float64) if not sp.issparse(X_comb) else X_comb.copy()
        if sp.issparse(X_comb_bin):
            X_comb_bin.data = (X_comb_bin.data > 0).astype(np.float64)
        X_comb_T = X_comb_bin.T.tocsc() if sp.issparse(X_comb_bin) else X_comb_bin.T
        idx1 = np.where(actual_labels == label_1)[0]
        idx2 = np.where(actual_labels == label_2)[0]
        sgs1_actual = _compute_core_sgs(X_comb_T[:, idx1],
                                        adata_comb.obs['cell_senescence_bin'].values[idx1].astype(int), n_bins=n_bins)
        sgs2_actual = _compute_core_sgs(X_comb_T[:, idx2],
                                        adata_comb.obs['cell_senescence_bin'].values[idx2].astype(int), n_bins=n_bins)
        delta_actual = sgs1_actual - sgs2_actual
    elif level == "sample":
        sample_info = adata_sc.obs.groupby([groupby, sample_col], observed=True).size().reset_index(name='count')
        valid_samples_1 = \
        sample_info[(sample_info[groupby] == label_1) & (sample_info['count'] >= min_cells_per_sample)][
            sample_col].tolist()
        valid_samples_2 = \
        sample_info[(sample_info[groupby] == label_2) & (sample_info['count'] >= min_cells_per_sample)][
            sample_col].tolist()
        print(f"Valid samples for {label_1} (>= {min_cells_per_sample} cells): {len(valid_samples_1)}")
        print(f"Valid samples for {label_2} (>= {min_cells_per_sample} cells): {len(valid_samples_2)}")
        if len(valid_samples_1) < min_samples_for_sample or len(valid_samples_2) < min_samples_for_sample:
            raise ValueError(
                f"Insufficient samples for T-test. Require at least {min_samples_for_sample} valid samples per condition.")
        valid_all_samples = valid_samples_1 + valid_samples_2
        mask_comb = adata_sc.obs[sample_col].isin(valid_all_samples)
        adata_comb = adata_sc[mask_comb].copy()
        X_comb = adata_comb.X
        X_comb_bin = (X_comb > 0).astype(np.float64) if not sp.issparse(X_comb) else X_comb.copy()
        if sp.issparse(X_comb_bin): X_comb_bin.data = (X_comb_bin.data > 0).astype(np.float64)
        X_comb_T = X_comb_bin.T.tocsc() if sp.issparse(X_comb_bin) else X_comb_bin.T
        sample_sgs_dict = {}
        for samp in valid_all_samples:
            samp_idx = np.where(adata_comb.obs[sample_col] == samp)[0]
            samp_scores = adata_comb.obs[score_col].values[samp_idx]
            try:
                samp_bins = pd.qcut(samp_scores, q=n_bins, labels=False)
            except ValueError:
                samp_bins = pd.qcut(pd.Series(samp_scores).rank(level='first'), q=n_bins, labels=False).values
            sample_sgs_dict[samp] = _compute_core_sgs(X_comb_T[:, samp_idx], samp_bins.astype(int), n_bins=n_bins)

        sgs_matrix_1 = np.array([sample_sgs_dict[s] for s in valid_samples_1])
        sgs_matrix_2 = np.array([sample_sgs_dict[s] for s in valid_samples_2])
        sgs1_actual = np.mean(sgs_matrix_1, axis=0)
        sgs2_actual = np.mean(sgs_matrix_2, axis=0)
        delta_actual = sgs1_actual - sgs2_actual
    else:
        raise ValueError("Invalid method. Choose 'cell' or 'sample'.")

    # Common Gene Filtering
    if min_delta_abs and min_delta_abs > 0:
        positive_mask = (sgs1_actual > min_sgs_threshold) | (sgs2_actual > min_sgs_threshold)
        delta_mask = np.abs(delta_actual) >= min_delta_abs
        keep_mask = positive_mask & delta_mask
        keep_genes_idx = np.where(keep_mask)[0]
        print(f"Gene baseline filtering: {keep_mask.sum()}/{len(var_idx_all)} genes retained")
    else:
        keep_genes_idx = np.arange(len(var_idx_all))
        print("No baseline gene filtering applied.")

    # Condition filtering
    if condition_filter != "both":
        delta_filtered = delta_actual[keep_genes_idx]
        if condition_filter == label_1:
            cond_mask = delta_filtered > 0
        elif condition_filter == label_2:
            cond_mask = delta_filtered < 0
        keep_genes_idx = keep_genes_idx[cond_mask]
        print(f"Condition filtering ('{condition_filter}'): {len(keep_genes_idx)} genes retained")
    if len(keep_genes_idx) == 0:
        print("No genes left after filtering. Exiting.")
        return pd.DataFrame()

    # Calculate P-values based on level
    delta_actual_filtered = delta_actual[keep_genes_idx]
    if level == "cell":
        X_perm = X_comb_T[keep_genes_idx]
        null_diff_counts = np.zeros(len(keep_genes_idx))
        for i in range(n_perms):
            shuffled_labels = np.random.permutation(actual_labels)
            idx1_perm = np.where(shuffled_labels == label_1)[0]
            idx2_perm = np.where(shuffled_labels == label_2)[0]
            shuffled_bins = np.zeros_like(actual_labels, dtype=int)
            for j, cond in enumerate([label_1, label_2]):
                cond_idx = np.where(shuffled_labels == cond)[0]
                cond_scores = adata_comb.obs[score_col].values[cond_idx]
                try:
                    cond_bins = pd.qcut(cond_scores, q=n_bins, labels=False)
                except ValueError:
                    cond_bins = pd.qcut(pd.Series(cond_scores).rank(level='first'), q=n_bins, labels=False).values
                shuffled_bins[cond_idx] = cond_bins
            sgs_perm1 = _compute_core_sgs(X_perm[:, idx1_perm], shuffled_bins[idx1_perm], n_bins=n_bins)
            sgs_perm2 = _compute_core_sgs(X_perm[:, idx2_perm], shuffled_bins[idx2_perm], n_bins=n_bins)
            delta_perm = sgs_perm1 - sgs_perm2
            null_diff_counts += (np.abs(delta_perm) >= np.abs(delta_actual_filtered))
        p_values = (null_diff_counts + 1) / (n_perms + 1)
    elif level == "sample":
        p_values = stats.ttest_ind(sgs_matrix_1[:, keep_genes_idx],
                                   sgs_matrix_2[:, keep_genes_idx],
                                   axis=0, equal_var=False)[1]
    _, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    dynamic_trends = np.where(delta_actual_filtered > 0, "Stronger", "Weaker")
    result_df = pd.DataFrame({
        'Gene': var_idx_all[keep_genes_idx],
        f'SGS_{label_1}': sgs1_actual[keep_genes_idx],
        f'SGS_{label_2}': sgs2_actual[keep_genes_idx],
        'Delta_SGS': delta_actual_filtered,
        'P_value': p_values,
        'FDR': p_adj,
        'Dynamic_Trend': dynamic_trends
    })
    result_df = result_df[result_df['FDR'] < 0.05].copy()
    print(f"Analysis complete: {len(result_df)} significant genes identified (FDR < 0.05).")
    if not result_df.empty:
        trend_counts = result_df['Dynamic_Trend'].value_counts()
        print(" | ".join(
            f"{trend}: {trend_counts[trend]}"
            for trend in ["Stronger", "Weaker"]
            if trend in trend_counts.index
        ))
    result_df = result_df.set_index("Gene")
    result_df.index.name = "Gene"
    return result_df