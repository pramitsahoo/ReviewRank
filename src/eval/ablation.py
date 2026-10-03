import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.ranking.ranker import MLPRanker, SparseEmbedding
from src.eval.metrics import recall_at_k


# ──────────────────────────────────────────────────────────────
# 1. Ranking model ablation
# ──────────────────────────────────────────────────────────────
class DenseOnlyRanker(nn.Module):
    def __init__(self, dense_feature_dim, mlp_dims=[256, 128], dropout=0.2):
        super().__init__()
        layers = []
        in_dim = dense_feature_dim
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
        return self.mlp(dense_features).squeeze(1)


class ShallowRanker(nn.Module):
    def __init__(self, sparse_feature_dims, dense_feature_dim, embed_dim=16):
        super().__init__()
        self.embedding = SparseEmbedding(sparse_feature_dims, embed_dim)
        in_dim = len(sparse_feature_dims) * embed_dim + dense_feature_dim
        self.linear = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, sparse_features, dense_features):
        emb = self.embedding(sparse_features)
        x = torch.cat([emb, dense_features], dim=1)
        return self.linear(x).squeeze(1)


def train_and_eval_ranker(model, train_loader, val_loader, device,
                          lr=0.001, epochs=10, patience=3, name=""):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for sparse, dense, label in tqdm(train_loader, desc=f"  {name} epoch {epoch}", leave=False):
            sparse = {k: v.to(device) for k, v in sparse.items()}
            dense, label = dense.to(device), label.to(device)
            loss = criterion(model(sparse, dense), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for sparse, dense, label in val_loader:
                sparse = {k: v.to(device) for k, v in sparse.items()}
                dense = dense.to(device)
                pred = torch.sigmoid(model(sparse, dense)).cpu().numpy()
                preds.append(pred)
                labels.append(label.numpy())

        all_labels = np.concatenate(labels)
        if len(np.unique(all_labels)) < 2:
            continue
        auc = roc_auc_score(all_labels, np.concatenate(preds))

        if auc > best_auc:
            best_auc = auc
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return best_auc


def ranking_ablation(sparse_feature_dims, dense_feature_dim, embed_dim,
                     mlp_dims, dropout, train_loader, val_loader, device,
                     lr=0.001, epochs=10, patience=3):
    results = {}

    full = MLPRanker(sparse_feature_dims, dense_feature_dim, embed_dim, mlp_dims, dropout)
    results["Full"] = train_and_eval_ranker(full, train_loader, val_loader, device, lr, epochs, patience, name="Full")

    dense_only = DenseOnlyRanker(dense_feature_dim, mlp_dims, dropout)
    results["DenseOnly"] = train_and_eval_ranker(dense_only, train_loader, val_loader, device, lr, epochs, patience, name="DenseOnly")

    shallow = ShallowRanker(sparse_feature_dims, dense_feature_dim, embed_dim)
    results["Shallow"] = train_and_eval_ranker(shallow, train_loader, val_loader, device, lr, epochs, patience, name="Shallow")

    return results


# ──────────────────────────────────────────────────────────────
# 2. Retrieval ablation
# ──────────────────────────────────────────────────────────────
def _random_recall(test_positive_items, all_item_ids, top_k, k):
    recalls = []
    all_items_list = all_item_ids.tolist()
    for pos in test_positive_items:
        retrieved = random.sample(all_items_list, min(top_k, len(all_items_list)))
        recalls.append(recall_at_k([pos], retrieved, k))
    return float(np.mean(recalls))


def _popularity_recall(test_positive_items, train_item_counts, top_k, k):
    popular_items = [item for item, _ in train_item_counts.most_common(top_k)]
    recalls = []
    for pos in test_positive_items:
        recalls.append(recall_at_k([pos], popular_items, k))
    return float(np.mean(recalls))


@torch.no_grad()
def retrieval_ablation(retrieval_model, faiss_index, test_loader, device,
                       all_item_ids, train_df, top_k=200, k=200):

    retrieval_model.eval()
    user_tower = retrieval_model.user_tower

    all_user_embeds = []
    all_positive_items = []
    for user_id, user_history, item_id, category_id in tqdm(test_loader, desc="Retrieval ablation", leave=False):
        user_emb = user_tower(user_id.to(device), user_history.to(device)).cpu().numpy()
        all_user_embeds.append(user_emb)
        all_positive_items.append(item_id.numpy())

    all_user_embeds = np.vstack(all_user_embeds).astype(np.float32)
    all_positive_items = np.concatenate(all_positive_items)

    _, retrieved = faiss_index.search(all_user_embeds, top_k)
    tt_recalls = []
    for i in range(len(all_positive_items)):
        tt_recalls.append(recall_at_k([all_positive_items[i]], retrieved[i].tolist(), k))
    tt_recall = float(np.mean(tt_recalls))

    random_recall = _random_recall(all_positive_items.tolist(), all_item_ids, top_k, k)

    train_item_counts = Counter(train_df["item_id"].tolist())
    pop_recall = _popularity_recall(all_positive_items.tolist(), train_item_counts, top_k, k)

    return {
        "TwoTower+FAISS": tt_recall,
        "Random": random_recall,
        "Popularity": pop_recall,
    }


# ──────────────────────────────────────────────────────────────
# 3. Feature ablation
# ──────────────────────────────────────────────────────────────
def _mask_features(loader, mask_user=False, mask_item=False, mask_cross=False):
    for sparse, dense, label in loader:
        dense = dense.clone()
        if mask_user:
            dense[:, 0:4] = 0.0
        if mask_item:
            dense[:, 4:7] = 0.0
        if mask_cross:
            dense[:, 7:9] = 0.0
        yield sparse, dense, label


@torch.no_grad()
def feature_ablation(model, val_loader, device):
    model.eval()

    def _eval(data_iter):
        preds, labels = [], []
        for sparse, dense, label in data_iter:
            sparse = {k: v.to(device) for k, v in sparse.items()}
            dense = dense.to(device)
            pred = torch.sigmoid(model(sparse, dense)).cpu().numpy()
            preds.append(pred)
            labels.append(label.numpy())
        all_labels = np.concatenate(labels)
        if len(np.unique(all_labels)) < 2:
            return 0.0
        return roc_auc_score(all_labels, np.concatenate(preds))

    results = {}
    results["Full"] = _eval(val_loader)
    results["NoUser"] = _eval(_mask_features(val_loader, mask_user=True))
    results["NoItem"] = _eval(_mask_features(val_loader, mask_item=True))
    results["NoCross"] = _eval(_mask_features(val_loader, mask_cross=True))
    return results
