import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import onnxruntime as ort

from src.retrieval.two_tower import TwoTowerModel
from src.retrieval.index import FAISSIndex
from src.data.vocab import load_vocab
from src.data.features import build_user_features, build_item_features
from src.utils import load_config, vocab_sizes, build_item_category_arrays, build_user_cat_stats


class RecommendationEngine:
    def __init__(self, cfg):
        device = torch.device("cpu")
        self.device = device

        processed_dir = Path(cfg["data"]["processed_dir"])
        artifacts_dir = Path(cfg["artifacts_dir"])

        # load vocab and data
        self.vocab = load_vocab(str(processed_dir / "vocab.json"))
        train_df = pd.read_parquet(processed_dir / "train.parquet")
        num_users, num_items, num_categories = vocab_sizes(self.vocab)

        self.user_features = build_user_features(train_df)

        meta_path = processed_dir / "meta.parquet"
        meta_df = pd.read_parquet(meta_path) if meta_path.exists() else None
        self.item_features = build_item_features(train_df, meta_df) if meta_df is not None else None

        self._user_cat_stats = build_user_cat_stats(train_df, meta_df) if meta_df is not None else {}

        # reverse vocab
        self.id_to_raw_item = {v: k for k, v in self.vocab["item_id"].items()}

        # load user tower from TwoTowerModel checkpoint
        ret_cfg = cfg["retrieval"]
        retrieval_model = TwoTowerModel(
            num_users=num_users, num_items=num_items, num_categories=num_categories,
            embedding_dim=ret_cfg["embedding_dim"],
            user_mlp_dims=ret_cfg["mlp_dims"],
            item_mlp_dims=[ret_cfg["embedding_dim"]],
            temperature=ret_cfg["temperature"],
        )
        ckpt = torch.load(artifacts_dir / "retriever_best.pt", map_location=device, weights_only=True)
        retrieval_model.load_state_dict(ckpt["model_state_dict"])
        retrieval_model.eval()
        self.user_tower = retrieval_model.user_tower

        # build user history lookup from training data
        self.max_history_length = cfg["data"]["max_history_length"]
        self.user_histories = {}
        sorted_df = train_df.sort_values("timestamp")
        for user_raw, group in sorted_df.groupby("user_id"):
            uid = self.vocab["user_id"].get(str(user_raw), 0)
            items = [self.vocab["item_id"].get(str(i), 0) for i in group["item_id"]]
            self.user_histories[uid] = items

        # load FAISS index
        idx_cfg = cfg["index"]
        self.retrieval_top_k = idx_cfg["top_k"]
        self.faiss_index = FAISSIndex(
            embedding_dim=ret_cfg["embedding_dim"],
            index_type=idx_cfg["index_type"],
            nprobe=idx_cfg["nprobe"],
        )
        index_path = str(artifacts_dir / "faiss.index")
        if Path(index_path).exists():
            self.faiss_index.load(index_path)
        else:
            all_item_ids, all_category_ids = build_item_category_arrays(train_df, self.vocab)
            self.faiss_index.build(
                item_tower=retrieval_model.item_tower,
                all_item_ids=all_item_ids,
                all_category_ids=all_category_ids,
                device=device,
                save_path=index_path,
            )

        # load ONNX ranker
        onnx_path = str(artifacts_dir / "ranker.onnx")
        self.ort_session = ort.InferenceSession(onnx_path)

        self.known_users = set(self.vocab["user_id"].keys())

    def _encode_user(self, user_id):
        uid = self.vocab["user_id"].get(user_id, 0)
        history = self.user_histories.get(uid, [])
        history = history[-self.max_history_length:]
        pad_len = self.max_history_length - len(history)
        history_padded = [0] * pad_len + history

        with torch.no_grad():
            uid_t = torch.tensor([uid], dtype=torch.long)
            hist_t = torch.tensor([history_padded], dtype=torch.long)
            emb = self.user_tower(uid_t, hist_t).numpy()
        return emb.astype(np.float32)

    def _build_onnx_inputs(self, user_id, candidate_encoded):
        B = len(candidate_encoded)
        uid_enc = self.vocab["user_id"].get(user_id, 0)

        # user dense features
        if user_id in self.user_features.index:
            uf = self.user_features.loc[user_id]
            u_dense = [float(uf.get("avg_rating", 0.0)), float(uf.get("num_interactions", 0.0)),
                       float(uf.get("std_rating", 0.0)), float(uf.get("active_days", 0.0))]
        else:
            u_dense = [0.0, 0.0, 0.0, 0.0]

        user_ids = np.full(B, uid_enc, dtype=np.int64)
        item_ids = np.zeros(B, dtype=np.int64)
        cat_ids = np.zeros(B, dtype=np.int64)
        dense = np.zeros((B, 9), dtype=np.float32)

        for i, iid_enc in enumerate(candidate_encoded):
            item_ids[i] = iid_enc
            iid_raw = self.id_to_raw_item.get(iid_enc, "")

            if self.item_features is not None and iid_raw in self.item_features.index:
                itf = self.item_features.loc[iid_raw]
                cat_ids[i] = self.vocab["category"].get(str(itf.get("category", "")), 0)
                i_dense = [float(itf.get("avg_rating", 0.0)), float(itf.get("num_ratings", 0.0)),
                           float(itf.get("price", 0.0))]
                cat_raw = str(itf.get("category", ""))
            else:
                i_dense = [0.0, 0.0, 0.0]
                cat_raw = ""

            # cross features
            avg = self._user_cat_stats.get((user_id, cat_raw))
            cross = [1.0, avg] if avg is not None else [0.0, 0.0]

            dense[i] = u_dense + i_dense + cross

        return {
            "user_id": user_ids,
            "item_id": item_ids,
            "category_id": cat_ids,
            "dense_features": dense,
        }

    # full recommendation pipeline
    def recommend(self, user_id, top_k = 20):
        t_start = time.perf_counter()

        # encode user and retrieve candidates via FAISS
        t0 = time.perf_counter()
        user_vec = self._encode_user(user_id)
        _, candidate_ids = self.faiss_index.search(user_vec, self.retrieval_top_k)
        candidate_encoded = candidate_ids[0].tolist()
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # build features and score with ONNX ranker
        t0 = time.perf_counter()
        onnx_inputs = self._build_onnx_inputs(user_id, candidate_encoded)
        logits = self.ort_session.run(None, onnx_inputs)[0]
        scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        ranking_ms = (time.perf_counter() - t0) * 1000

        # sort and take top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            raw_id = self.id_to_raw_item.get(candidate_encoded[idx], str(candidate_encoded[idx]))
            recommendations.append({
                "item_id": raw_id,
                "score": float(scores[idx]),
                "rank": rank,
            })

        total_ms = (time.perf_counter() - t_start) * 1000

        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "retrieval_latency_ms": round(retrieval_ms, 2),
            "ranking_latency_ms": round(ranking_ms, 2),
            "total_latency_ms": round(total_ms, 2),
        }
