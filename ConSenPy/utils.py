import torch
import numpy as np
import torch.nn.functional as F
import copy
import gc
import scipy.stats as stats
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from .model_architecture import SIMPLE_VAE


def get_spatial_graph(adata, n_neighbors=5, device="cpu"):
    coor = adata.obsm["spatial"]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    indices = indices[:, 1:]
    n = indices.shape[0]
    row = np.repeat(np.arange(n), n_neighbors)
    col = indices.flatten()
    weights = np.ones(len(row), dtype=np.float32)
    adj = sp.coo_matrix((weights, (row, col)), shape=(n, n))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0], dtype=np.float32)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    adj_norm = adj_norm.tocoo()
    indices_torch = torch.tensor(
        np.vstack((adj_norm.row, adj_norm.col)),
        dtype=torch.long
    )
    values_torch = torch.tensor(adj_norm.data, dtype=torch.float32)
    adj_tensor = torch.sparse_coo_tensor(indices_torch, values_torch, adj_norm.shape).coalesce()
    return adj_tensor.to(device)

class VAE_Trainer:
    def __init__(self, epochs, z_dim, layer_dims, lr, dropout, device, seed, patience=20, is_spatial=False):
        self.epochs = epochs
        self.z_dim = z_dim
        self.layer_dims = layer_dims
        self.base_lr = lr
        self.dropout = dropout
        self.device = device
        self.seed = seed
        self.patience = patience
        self.is_spatial = is_spatial
    def initialize(self, adata, batch_key="batch"):
        self.input_dim = adata.shape[1]
        X_raw = adata.layers["counts"] if "counts" in adata.layers else adata.X.copy()
        X_norm = adata.X
        if sp.issparse(X_raw):
            lib_size = np.array(X_raw.sum(1)).flatten()
            X_raw = X_raw.toarray()
        else:
            lib_size = np.sum(X_raw, axis=1)
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()
        self.X_raw = torch.FloatTensor(X_raw)
        self.X_norm = torch.FloatTensor(X_norm)
        self.lib_size = torch.FloatTensor(lib_size).reshape(-1, 1)
        del X_raw, X_norm, lib_size
        gc.collect()
        if batch_key not in adata.obs:
            adata.obs[batch_key] = 'batch_1'
        batch_values = adata.obs[batch_key].astype('category')
        self.batch_indices = torch.tensor(batch_values.cat.codes.values, dtype=torch.long)
        self.batch_dim = len(batch_values.cat.categories)
        self.adj = None
        if self.is_spatial:
            print("Spatial mode ON. Building spatial graph...")
            self.adj = get_spatial_graph(adata, n_neighbors=5, device=self.device)
        self.model = SIMPLE_VAE(
            input_dim=self.input_dim,
            batch_dim=self.batch_dim,
            z_dim=self.z_dim,
            layer_dims=self.layer_dims,
            seed=self.seed,
            dropout=self.dropout,
            is_spatial=self.is_spatial
        ).to(self.device)

    def train(self):
        best_loss = float('inf')
        patience_counter = 0
        rel_delta = 1e-3
        n_cells = self.X_raw.size(0)
        n_genes = self.X_raw.size(1)
        if n_cells <= 1024:
            batch_size = n_cells
        elif n_cells < 10000:
            batch_size = 128
        elif n_cells < 50000:
            batch_size = 256
        elif n_cells < 100000:
            batch_size = 512
        else:
            batch_size = 1024
        scale_factor = (batch_size / 128) ** 0.5 if not self.is_spatial else 1.0
        dynamic_lr = self.base_lr * scale_factor
        print(f"Dataset Size: {n_cells} cells, {n_genes} genes.")
        if self.is_spatial:
            print(f"Training Mode: Full-batch GCN, Initial LR: {dynamic_lr:.1e}")
        else:
            print(f"Training Mode: Mini-batch Auto, Batch Size: {batch_size}, Initial LR: {dynamic_lr:.1e}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=dynamic_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5,
            threshold=rel_delta, threshold_mode='rel', min_lr=1e-5
        )
        best_model_wts = copy.deepcopy(self.model.state_dict())
        if not self.is_spatial:
            dataset = TensorDataset(self.X_raw, self.X_norm, self.lib_size, self.batch_indices)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        kl_warmup_epochs = max(1, self.epochs // 3)
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
            kl_weight = min(1.0, epoch / kl_warmup_epochs)
            if self.is_spatial:
                batch_raw = self.X_raw.to(self.device)
                batch_norm = self.X_norm.to(self.device)
                batch_lib = self.lib_size.to(self.device)
                batch_idx = self.batch_indices.to(self.device)
                batch_one_hot = F.one_hot(batch_idx, num_classes=self.batch_dim).float()
                mu_dec, drop_log, mu_enc, logvar = self.model(batch_norm, batch_lib, batch_one_hot, self.adj)
                loss, recon, kl = self.model.loss_function(batch_raw, mu_dec, drop_log, mu_enc, logvar,kl_weight=kl_weight)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.optimizer.step()
                epoch_loss = loss.item() * n_cells
                epoch_recon = recon.item() * n_cells
                epoch_kl = kl.item() * n_cells
            else:
                for batch_raw, batch_norm, batch_lib, batch_idx in dataloader:
                    batch_raw = batch_raw.to(self.device)
                    batch_norm = batch_norm.to(self.device)
                    batch_lib = batch_lib.to(self.device)
                    batch_idx = batch_idx.to(self.device)
                    batch_one_hot = F.one_hot(batch_idx, num_classes=self.batch_dim).float()
                    mu_dec, drop_log, mu_enc, logvar = self.model(batch_norm, batch_lib, batch_one_hot)
                    loss, recon, kl = self.model.loss_function(batch_raw, mu_dec, drop_log, mu_enc, logvar,kl_weight=kl_weight)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                    self.optimizer.step()
                    batch_len = batch_raw.size(0)
                    epoch_loss += loss.item() * batch_len
                    epoch_recon += recon.item() * batch_len
                    epoch_kl += kl.item() * batch_len
            avg_eval_loss = (epoch_recon + epoch_kl) / n_cells
            scheduler.step(avg_eval_loss)
            if best_loss == float('inf') or avg_eval_loss < best_loss - abs(best_loss) * rel_delta:
                best_loss = avg_eval_loss
                patience_counter = 0
                best_model_wts = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
            if epoch % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                avg_weighted_loss = epoch_loss / n_cells
                print(
                    f"Epoch {epoch:4d} | KL_wt: {kl_weight:.2f} | Loss: {avg_weighted_loss:.2f} | Eval ELBO: {avg_eval_loss:.2f} | LR: {current_lr:.1e}")
            if patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Eval ELBO: {best_loss:.2f}")
                break
        self.model.load_state_dict(best_model_wts)
        print("Training finished. Best model weights restored.")

    def get_latent(self):
        self.model.eval()
        Z_list = []
        with torch.no_grad():
            if self.is_spatial:
                batch_norm = self.X_norm.to(self.device)
                batch_idx = self.batch_indices.to(self.device)
                batch_one_hot = F.one_hot(batch_idx, num_classes=self.batch_dim).float()
                x_log = torch.log1p(batch_norm)
                mu_enc, logvar_enc = self.model.encoder(x_log, batch_one_hot, self.adj)
                Z_list.append(mu_enc.cpu().numpy())
            else:
                batch_size = 2048
                dataset = TensorDataset(self.X_norm, self.lib_size, self.batch_indices)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                for batch_norm, batch_lib, batch_idx in dataloader:
                    batch_norm = batch_norm.to(self.device)
                    batch_one_hot = F.one_hot(batch_idx.to(self.device), num_classes=self.batch_dim).float()
                    x_log = torch.log1p(batch_norm)
                    mu_enc, logvar_enc = self.model.encoder(x_log, batch_one_hot)
                    Z_list.append(mu_enc.cpu().numpy())
        return np.vstack(Z_list)


def compute_bulk_similarity_score(adata_st, adata_bulk, bulk_labels):
    """
    compute bulk similarity score (Spearman correlation based)
    """
    common_genes = sorted(set(adata_st.var_names).intersection(adata_bulk.var_names))
    adata_st = adata_st[:, common_genes].copy()
    adata_bulk = adata_bulk[:, common_genes].copy()
    st_indices = [adata_st.var_names.get_loc(g) for g in common_genes]
    bulk_indices = [adata_bulk.var_names.get_loc(g) for g in common_genes]
    X_st_raw = np.log1p(adata_st.X)
    if sp.issparse(X_st_raw):
        X_st = X_st_raw[:, st_indices].toarray()
    else:
        X_st = X_st_raw[:, st_indices]
    X_bulk_raw = adata_bulk.X
    if sp.issparse(X_bulk_raw):
        X_bulk = X_bulk_raw[:, bulk_indices].toarray()
    else:
        X_bulk = X_bulk_raw[:, bulk_indices]
    X_bulk = np.log1p(X_bulk)
    sen_mask = bulk_labels == 1
    non_mask = bulk_labels == 0
    sen_bulk_profile = X_bulk[sen_mask, :].mean(axis=0)
    non_bulk_profile = X_bulk[non_mask, :].mean(axis=0)
    rank_st = stats.rankdata(X_st, axis=1)
    rank_sen = stats.rankdata(sen_bulk_profile)
    rank_non = stats.rankdata(non_bulk_profile)
    rank_st_centered = rank_st - rank_st.mean(axis=1, keepdims=True)
    rank_sen_centered = rank_sen - rank_sen.mean()
    rank_non_centered = rank_non - rank_non.mean()
    cov_sen = rank_st_centered.dot(rank_sen_centered)
    cov_non = rank_st_centered.dot(rank_non_centered)
    std_st = np.sqrt(np.sum(rank_st_centered ** 2, axis=1))
    std_sen = np.sqrt(np.sum(rank_sen_centered ** 2))
    std_non = np.sqrt(np.sum(rank_non_centered ** 2))
    corr_sen = cov_sen / (std_st * std_sen + 1e-8)
    corr_non = cov_non / (std_st * std_non + 1e-8)
    s = corr_sen - corr_non
    s_normalized = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-8)

    return s_normalized

def logistic_regression_scoring(Z, s_score, top_percent=10, bottom_percent=10):
    high_threshold = np.percentile(s_score, 100 - top_percent)
    low_threshold = np.percentile(s_score, bottom_percent)
    high_idx = np.where(s_score >= high_threshold)[0]
    low_idx = np.where(s_score <= low_threshold)[0]
    X_train = np.vstack([Z[high_idx], Z[low_idx]])
    y_train = np.array([1] * len(high_idx) + [0] * len(low_idx))
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    coef = clf.coef_[0]
    Z_scaled = scaler.transform(Z)
    final_score = Z_scaled @ coef
    final_score = (final_score - final_score.min()) / (final_score.max() - final_score.min() + 1e-8)
    return final_score


def _compute_core_sgs(X_bin_T, bin_labels):
    n_samples = X_bin_T.shape[1]
    Y = sp.csr_matrix((np.ones(n_samples, dtype=np.float64),(np.arange(n_samples), bin_labels)),shape=(n_samples, 5))
    P_count = X_bin_T.dot(Y).toarray()
    bin_sizes = np.bincount(bin_labels, minlength=5)
    bin_sizes[bin_sizes == 0] = 1
    P = P_count / bin_sizes
    P_pct = P * 100
    p_young = P_pct[:, 0]
    p_old = P_pct[:, 4]

    # ========== old(x) ==========
    x_old = p_old
    val_old = np.zeros_like(x_old)
    mask1 = (x_old > 0) & (x_old <= 3)
    val_old[mask1] = 1.0 / (1.0 + np.exp(-2.0 * x_old[mask1]))
    mask2 = (x_old > 3) & (x_old <= 20)
    val_old[mask2] = 1.0
    mask3 = x_old > 20
    val_old[mask3] = -0.25 * x_old[mask3] + 6.0

    # ========== young(x) ==========
    x_young = p_young
    val_young = np.ones_like(x_young)
    mask = x_young >= 5
    val_young[mask] = -0.5 * x_young[mask] + 3.5

    # ========== gain(x) ==========
    x_gain = p_old - p_young
    val_gain = np.ones_like(x_gain)
    mask1 = x_gain < 5
    val_gain[mask1] = x_gain[mask1] / 5.0
    mask2 = x_gain > 15
    val_gain[mask2] = -0.2 * x_gain[mask2] + 4.0

    # ========== ratio(x) ==========
    p_young_safe = np.maximum(p_young, 0.01)
    ratio_raw = p_old / p_young_safe
    val_ratio = np.minimum(ratio_raw, 2.5)
    sgs = val_old + val_gain + val_young + val_ratio
    return sgs