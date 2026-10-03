from bisect import bisect_left

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils import build_user_cat_stats


class RetrievalDataset(Dataset):
    def __init__(self, df, vocab, max_history_length, history_df=None):
        self.vocab = vocab
        self.max_history_length = max_history_length

        # build per-user history as sorted (timestamps[], items[]) from train data only
        src = history_df if history_df is not None else df
        user_timestamps: dict[int, list[float]] = {}
        user_items: dict[int, list[int]] = {}
        for row in src.sort_values("timestamp").itertuples():
            uid = vocab["user_id"].get(str(row.user_id), 0)
            iid = vocab["item_id"].get(str(row.item_id), 0)
            user_timestamps.setdefault(uid, []).append(float(row.timestamp))
            user_items.setdefault(uid, []).append(iid)

        # build samples; use bisect to find history prefix in O(log n) per sample
        self.samples = []
        for row in tqdm(df.itertuples(), total=len(df), desc="Building retrieval samples",
                        leave=False, mininterval=1.0):
            uid = vocab["user_id"].get(str(row.user_id), 0)
            iid = vocab["item_id"].get(str(row.item_id), 0)
            cid = vocab["category"].get(str(row.category), 0)
            ts = float(row.timestamp)

            ts_list = user_timestamps.get(uid, [])
            cut = bisect_left(ts_list, ts)
            history = user_items.get(uid, [])[:cut]
            history = history[-max_history_length:]

            pad_len = max_history_length - len(history)
            history_padded = [0] * pad_len + history

            self.samples.append((uid, history_padded, iid, cid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        uid, history_padded, iid, cid = self.samples[idx]
        return (
            torch.tensor(uid, dtype=torch.long),
            torch.tensor(history_padded, dtype=torch.long),
            torch.tensor(iid, dtype=torch.long),
            torch.tensor(cid, dtype=torch.long),
        )


class RankingDataset(Dataset):
    def __init__(self, df, vocab, user_features, item_features, meta_df=None, train_df=None):
        self.vocab = vocab
        self.records = []

        user_cat_stats = build_user_cat_stats(train_df, meta_df) if meta_df is not None and train_df is not None else {}

        for row in tqdm(df.itertuples(), total=len(df), desc="Building ranking samples",
                        leave=False, mininterval=1.0):
            uid_raw = str(row.user_id)
            iid_raw = str(row.item_id)
            cat_raw = str(row.category)

            uid = vocab["user_id"].get(uid_raw, 0)
            iid = vocab["item_id"].get(iid_raw, 0)
            cid = vocab["category"].get(cat_raw, 0)

            # user dense features
            if uid_raw in user_features.index or row.user_id in user_features.index:
                key = uid_raw if uid_raw in user_features.index else row.user_id
                uf = user_features.loc[key]
                u_dense = [
                    float(uf.get("avg_rating", 0.0)),
                    float(uf.get("num_interactions", 0.0)),
                    float(uf.get("std_rating", 0.0)),
                    float(uf.get("active_days", 0.0)),
                ]
            else:
                u_dense = [0.0, 0.0, 0.0, 0.0]

            # item dense features
            if item_features is not None and (iid_raw in item_features.index or row.item_id in item_features.index):
                key = iid_raw if iid_raw in item_features.index else row.item_id
                itf = item_features.loc[key]
                i_dense = [
                    float(itf.get("avg_rating", 0.0)),
                    float(itf.get("num_ratings", 0.0)),
                    float(itf.get("price", 0.0)),
                ]
            else:
                i_dense = [0.0, 0.0, 0.0]

            # cross features via precomputed stats
            avg = user_cat_stats.get((row.user_id, cat_raw))
            cross = [1.0, avg] if avg is not None else [0.0, 0.0]

            dense = u_dense + i_dense + cross

            self.records.append({
                "sparse": {"user_id": uid, "item_id": iid, "category_id": cid},
                "dense": dense,
                "label": float(row.label),
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        sparse = {k: torch.tensor(v, dtype=torch.long) for k, v in rec["sparse"].items()}
        dense = torch.tensor(rec["dense"], dtype=torch.float32)
        label = torch.tensor(rec["label"], dtype=torch.float32)
        return sparse, dense, label
