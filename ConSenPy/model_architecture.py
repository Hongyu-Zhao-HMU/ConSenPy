import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.distributions import ZeroInflatedNegativeBinomial

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, adj):
        x = self.dropout(x)
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support) + self.bias
        return output

class VAE_GCNEncoder(nn.Module):
    def __init__(self, input_dim, batch_dim, z_dim, layer_dims, dropout=0.1):
        super().__init__()
        self.gcns = nn.ModuleList()
        in_dim = input_dim + batch_dim
        for dim in layer_dims:
            self.gcns.append(GCNLayer(in_dim, dim, dropout))
            in_dim = dim
        self.fc_mean = nn.Linear(in_dim, z_dim)
        self.fc_logvar = nn.Linear(in_dim, z_dim)
    def forward(self, x, batch_one_hot, adj):
        h = torch.cat([x, batch_one_hot], dim=-1)
        for gcn in self.gcns:
            h = F.leaky_relu(gcn(h, adj))
        return self.fc_mean(h), self.fc_logvar(h)

class Encoder(nn.Module):
    def __init__(self, input_dim, batch_dim, z_dim, layer_dims, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim + batch_dim
        for dim in layer_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = dim
        self.backbone = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(in_dim, z_dim)
        self.fc_logvar = nn.Linear(in_dim, z_dim)
    def forward(self, x, batch_one_hot):
        h = torch.cat([x, batch_one_hot], dim=-1)
        h = self.backbone(h)
        return self.fc_mean(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, input_dim, batch_dim, z_dim, layer_dims, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = z_dim + batch_dim
        for dim in layer_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = dim
        self.backbone = nn.Sequential(*layers)
        self.fc_log_mean = nn.Linear(in_dim, input_dim)
        self.fc_gate_logits = nn.Linear(in_dim, input_dim)
    def forward(self, z, batch_one_hot, library_size):
        h = torch.cat([z, batch_one_hot], dim=-1)
        h = self.backbone(h)
        log_mu = self.fc_log_mean(h) + torch.log(library_size + 1e-6)
        mu = torch.exp(log_mu)
        gate_logits = self.fc_gate_logits(h)
        return mu, gate_logits

class SIMPLE_VAE(nn.Module):
    def __init__(self, input_dim, batch_dim, z_dim, layer_dims, seed, dropout=0.1, is_spatial=False):
        super().__init__()
        torch.manual_seed(seed)
        self.z_dim = z_dim
        self.batch_dim = batch_dim
        self.is_spatial = is_spatial
        if self.is_spatial:
            self.encoder = VAE_GCNEncoder(input_dim, batch_dim, z_dim, layer_dims, dropout)
        else:
            self.encoder = Encoder(input_dim, batch_dim, z_dim, layer_dims, dropout)
        self.decoder = Decoder(input_dim, batch_dim, z_dim, layer_dims[::-1], dropout)
        self.log_theta = nn.Parameter(torch.randn(input_dim))
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu
    def forward(self, x_normalized, library_size, batch_one_hot, adj=None):
        x_log = torch.log1p(x_normalized)
        if self.is_spatial:
            mu_enc, logvar_enc = self.encoder(x_log, batch_one_hot, adj)
        else:
            mu_enc, logvar_enc = self.encoder(x_log, batch_one_hot)
        z = self.reparameterize(mu_enc, logvar_enc)
        mu_dec, drop_log = self.decoder(z, batch_one_hot, library_size)
        return mu_dec, drop_log, mu_enc, logvar_enc

    def loss_function(self, x_raw, mu_dec, drop_log, mu_enc, logvar_enc, kl_weight=1.0):
        theta = F.softplus(self.log_theta) + 1e-6
        zinb_logits = torch.log(mu_dec + 1e-8) - torch.log(theta + 1e-8)
        zinb = ZeroInflatedNegativeBinomial(
            total_count=theta,
            logits=zinb_logits,
            gate_logits=drop_log,
            validate_args=False
        )
        recon_loss = -zinb.log_prob(x_raw).sum(dim=1).mean()
        kl_loss = -0.5 * torch.sum(
            1 + logvar_enc - mu_enc.pow(2) - logvar_enc.exp(),
            dim=1
        ).mean()
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss