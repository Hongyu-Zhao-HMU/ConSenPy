import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
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
    adata_sc.obs["SenseScore"] = final_score
    return adata_sc


def calculate_differential_sgs(adata_sc, groupby="Condition", label_1="Cond.1", label_2="Cond.2",
                               score_col="SenseScore", n_perms=500, min_delta_abs=1,
                               min_sgs_threshold=1, condition_filter="both"):
    """
       CAS-genes (Condition-Associated Senescence genes) Analysis
    """
    np.random.seed(42)
    print(f"--- Differential SGS Analysis: {label_1} vs {label_2} ---")
    mask_comb = (adata_sc.obs[groupby] == label_1) | (adata_sc.obs[groupby] == label_2)
    adata_comb = adata_sc[mask_comb].copy()
    adata_comb.obs['global_senescence_bin'] = -1
    for condition in [label_1, label_2]:
        cond_mask = adata_comb.obs[groupby] == condition
        cond_scores = adata_comb.obs.loc[cond_mask, score_col].values
        try:
            cond_bins = pd.qcut(cond_scores, q=5, labels=False)
        except ValueError:
            cond_bins = pd.qcut(pd.Series(cond_scores).rank(method='first'), q=5, labels=False).values
        adata_comb.obs.loc[cond_mask, 'global_senescence_bin'] = cond_bins
    actual_labels = adata_comb.obs[groupby].values
    X_comb = adata_comb.X
    X_comb_bin = (X_comb > 0).astype(np.float64) if not sp.issparse(X_comb) else X_comb.copy()
    if sp.issparse(X_comb_bin):
        X_comb_bin.data = (X_comb_bin.data > 0).astype(np.float64)
    X_comb_T = X_comb_bin.T.tocsc() if sp.issparse(X_comb_bin) else X_comb_bin.T
    idx1 = np.where(actual_labels == label_1)[0]
    idx2 = np.where(actual_labels == label_2)[0]
    sgs1_actual = _compute_core_sgs(X_comb_T[:, idx1], adata_comb.obs['global_senescence_bin'].values[idx1].astype(int))
    sgs2_actual = _compute_core_sgs(X_comb_T[:, idx2], adata_comb.obs['global_senescence_bin'].values[idx2].astype(int))

    delta_actual = sgs1_actual - sgs2_actual
    # Gene filtering
    var_idx_all = adata_comb.var_names
    if min_delta_abs and min_delta_abs > 0:
        positive_mask = (sgs1_actual > min_sgs_threshold) | (sgs2_actual > min_sgs_threshold)
        delta_mask = np.abs(delta_actual) >= min_delta_abs
        keep_mask = positive_mask & delta_mask
        keep_genes_idx = np.where(keep_mask)[0]
        print(f"Gene filtering: {keep_mask.sum()}/{len(var_idx_all)} genes retained")
    else:
        keep_genes_idx = np.arange(len(var_idx_all))
        print("No gene filtering applied.")

    # Condition filtering (before permutation test)
    if condition_filter != "both":
        delta_filtered = delta_actual[keep_genes_idx]
        if condition_filter == label_1:
            cond_mask = delta_filtered > 0
        elif condition_filter == label_2:
            cond_mask = delta_filtered < 0
        keep_genes_idx = keep_genes_idx[cond_mask]
        print(f"Condition filtering ('{condition_filter}'): {len(keep_genes_idx)} genes retained")

    # Permutation test on retained genes
    X_perm = X_comb_T[keep_genes_idx]
    null_diff_counts = np.zeros(len(keep_genes_idx))
    delta_actual_filtered = delta_actual[keep_genes_idx]
    for i in range(n_perms):
        shuffled_labels = np.random.permutation(actual_labels)
        idx1_perm = np.where(shuffled_labels == label_1)[0]
        idx2_perm = np.where(shuffled_labels == label_2)[0]
        shuffled_bins = np.zeros_like(actual_labels, dtype=int)
        for j, cond in enumerate([label_1, label_2]):
            cond_idx = np.where(shuffled_labels == cond)[0]
            cond_scores = adata_comb.obs[score_col].values[cond_idx]
            try:
                cond_bins = pd.qcut(cond_scores, q=5, labels=False)
            except ValueError:
                cond_bins = pd.qcut(pd.Series(cond_scores).rank(method='first'), q=5, labels=False).values
            shuffled_bins[cond_idx] = cond_bins
        sgs_perm1 = _compute_core_sgs(X_perm[:, idx1_perm], shuffled_bins[idx1_perm])
        sgs_perm2 = _compute_core_sgs(X_perm[:, idx2_perm], shuffled_bins[idx2_perm])
        delta_perm = sgs_perm1 - sgs_perm2
        null_diff_counts += (np.abs(delta_perm) >= np.abs(delta_actual_filtered))
    p_values = (null_diff_counts + 1) / (n_perms + 1)
    _, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    # Determine condition
    if condition_filter == "both":
        conditions = np.full(len(keep_genes_idx), "NS", dtype=object)
        for k in range(len(keep_genes_idx)):
            gene_idx = keep_genes_idx[k]
            if p_adj[k] < 0.05:
                if delta_actual[gene_idx] > 0:
                    conditions[k] = label_1
                elif delta_actual[gene_idx] < 0:
                    conditions[k] = label_2
    else:
        conditions = np.full(len(keep_genes_idx), "NS", dtype=object)
        conditions[p_adj < 0.05] = condition_filter

    # create DataFrame
    result_df = pd.DataFrame({
        'Gene': var_idx_all[keep_genes_idx],
        f'SGS_{label_1}': sgs1_actual[keep_genes_idx],
        f'SGS_{label_2}': sgs2_actual[keep_genes_idx],
        'Delta_SGS': delta_actual[keep_genes_idx],
        'Condition': conditions
    })
    result_df = result_df[result_df['Condition'] != "NS"]

    # print information
    sig_count = np.sum(p_adj < 0.05)
    print(f"Analysis complete: {len(var_idx_all)} genes total, {len(keep_genes_idx)} after filtering")
    if condition_filter == "both":
        count_1 = np.sum((p_adj < 0.05) & (conditions == label_1))
        count_2 = np.sum((p_adj < 0.05) & (conditions == label_2))
        print(f"Significant genes (FDR < 0.05): {sig_count}")
        print(f"  - Condition = {label_1}: {count_1}")
        print(f"  - Condition = {label_2}: {count_2}")
    else:
        print(f"Significant {condition_filter}-associated genes (FDR < 0.05): {sig_count}")
    return result_df