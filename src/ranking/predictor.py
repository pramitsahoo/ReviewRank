import torch
import numpy as np

from src.utils import build_user_cat_stats


class RankingPredictor:
    def __init__(self, model, vocab, user_features, item_features, device,
                 meta_df=None, train_df=None):
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.user_features = user_features
        self.item_features = item_features
        self.device = device
        self._user_cat_stats = build_user_cat_stats(train_df, meta_df) if meta_df is not None and train_df is not None else {}

    def _build_features(self, user_id, item_ids):
        B = len(item_ids)
        uid = self.vocab["user_id"].get(str(user_id), 0) # cold start: default to 0 (padding idx) if user_id not in vocab

        if user_id in self.user_features.index:
            uf = self.user_features.loc[user_id]
            u_dense = [float(uf.get("avg_rating", 0.0)), float(uf.get("num_interactions", 0.0)),
                       float(uf.get("std_rating", 0.0)), float(uf.get("active_days", 0.0))]
        else:
            u_dense = [0.0, 0.0, 0.0, 0.0]

        sparse_user = torch.full((B,), uid, dtype=torch.long)
        sparse_item = torch.zeros(B, dtype=torch.long)
        sparse_cat = torch.zeros(B, dtype=torch.long)
        dense_list = []

        for i, iid_raw in enumerate(item_ids):
            iid = self.vocab["item_id"].get(str(iid_raw), 0)
            sparse_item[i] = iid

            if self.item_features is not None and iid_raw in self.item_features.index:
                itf = self.item_features.loc[iid_raw]
                cid = self.vocab["category"].get(str(itf.get("category", "")), 0)
                i_dense = [float(itf.get("avg_rating", 0.0)), float(itf.get("num_ratings", 0.0)),
                           float(itf.get("price", 0.0))]
            else:
                cid = 0
                i_dense = [0.0, 0.0, 0.0]

            sparse_cat[i] = cid

            # cross features from precomputed stats
            cat_raw = str(itf.get("category", "")) if self.item_features is not None and iid_raw in self.item_features.index else ""
            avg = self._user_cat_stats.get((user_id, cat_raw))
            if avg is not None:
                cross = [1.0, avg]
            else:
                cross = [0.0, 0.0]
            dense_list.append(u_dense + i_dense + cross)

        sparse = {
            "user_id": sparse_user.to(self.device),
            "item_id": sparse_item.to(self.device),
            "category_id": sparse_cat.to(self.device),
        }
        dense = torch.tensor(dense_list, dtype=torch.float32).to(self.device)
        return sparse, dense

    @torch.no_grad()
    def predict(self, user_id, candidate_item_ids):
        sparse, dense = self._build_features(user_id, candidate_item_ids)
        logits = self.model(sparse, dense)
        scores = torch.sigmoid(logits).cpu().numpy()
        return scores

    def rank(self, user_id, candidate_item_ids, top_k):
        scores = self.predict(user_id, candidate_item_ids)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(candidate_item_ids[i], float(scores[i])) for i in top_indices]

    # batch prediction for evaluation
    @torch.no_grad()
    def predict_batch(self, user_ids, candidate_item_ids_list):
        all_sparse_user, all_sparse_item, all_sparse_cat, all_dense = [], [], [], []

        for user_id, item_ids in zip(user_ids, candidate_item_ids_list):
            sparse, dense = self._build_features(user_id, item_ids)
            all_sparse_user.append(sparse["user_id"])
            all_sparse_item.append(sparse["item_id"])
            all_sparse_cat.append(sparse["category_id"])
            all_dense.append(dense)

        sparse = {
            "user_id": torch.cat(all_sparse_user),
            "item_id": torch.cat(all_sparse_item),
            "category_id": torch.cat(all_sparse_cat),
        }
        dense = torch.cat(all_dense)

        logits = self.model(sparse, dense)
        scores = torch.sigmoid(logits).cpu().numpy()

        C = len(candidate_item_ids_list[0])
        return scores.reshape(-1, C)
