import torch
import torch.nn as nn


class SparseEmbedding(nn.Module):
    def __init__(self, feature_dims, embed_dim = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(num, embed_dim, padding_idx=0)
            for name, num in feature_dims.items()
        })
        self._init_weights()

    def _init_weights(self):
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
            nn.init.zeros_(emb.weight[0])  # padding idx

    def forward(self, sparse_features: dict[str, torch.Tensor]) -> torch.Tensor:
        embs = [self.embeddings[name](sparse_features[name]) for name in self.embeddings] # {"user_id": (B, embed_dim), "item_id": (B, embed_dim), ...}
        return torch.cat(embs, dim=-1) # (B, num_sparse_features * embed_dim)


class MLPRanker(nn.Module):
    def __init__(self, sparse_feature_dims, dense_feature_dim, embed_dim = 16, mlp_dims = [256, 128], dropout = 0.2):
        super().__init__()
        self.embedding = SparseEmbedding(sparse_feature_dims, embed_dim)

        in_dim = len(sparse_feature_dims) * embed_dim + dense_feature_dim
        layers = []
        for out_dim in mlp_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, sparse_features, dense_features):
        emb = self.embedding(sparse_features)
        x = torch.cat([emb, dense_features], dim=1)
        logit = self.mlp(x)
        return logit.squeeze(1) # raw logit, apply sigmoid externally
