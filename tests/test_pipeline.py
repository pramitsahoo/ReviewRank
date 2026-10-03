import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import load_and_filter, binarize_labels
from src.retrieval.two_tower import TwoTowerModel
from src.ranking.ranker import MLPRanker


# fixtures
@pytest.fixture
def sample_df():
    # 5 users, 5 items. user_5 and item_5 have only 1 interaction → should be filtered at k=2
    rows = []
    for u in range(1, 6):
        for i in range(1, 6):
            if u == 5 and i != 1:
                continue  # user_5 only interacts with item_1
            if i == 5 and u != 1:
                continue  # item_5 only interacted by user_1
            rows.append({
                "user_id": f"u{u}",
                "item_id": f"i{i}",
                "rating": float(np.random.randint(1, 6)),
                "timestamp": np.random.randint(1000000, 2000000),
            })
    return pd.DataFrame(rows)


# data preprocessing tests
class TestKCoreFiltering:
    def test_no_sparse_users_after_filtering(self, sample_df, tmp_path):
        sample_df.to_parquet(tmp_path / "reviews.parquet", index=False)

        k = 2
        df = load_and_filter(str(tmp_path), k_core=k)

        user_counts = df["user_id"].value_counts()
        assert (user_counts >= k).all(), f"Found users with < {k} interactions"

    def test_no_sparse_items_after_filtering(self, sample_df, tmp_path):
        sample_df.to_parquet(tmp_path / "reviews.parquet", index=False)

        k = 2
        df = load_and_filter(str(tmp_path), k_core=k)

        item_counts = df["item_id"].value_counts()
        assert (item_counts >= k).all(), f"Found items with < {k} interactions"

    def test_filtered_subset_of_original(self, sample_df, tmp_path):
        sample_df.to_parquet(tmp_path / "reviews.parquet", index=False)

        df = load_and_filter(str(tmp_path), k_core=2)
        assert len(df) <= len(sample_df)

    def test_binarize_labels(self, sample_df):
        df = binarize_labels(sample_df, threshold=4)
        assert "label" in df.columns
        assert set(df["label"].unique()).issubset({0, 1})
        assert (df.loc[df["rating"] >= 4, "label"] == 1).all()
        assert (df.loc[df["rating"] < 4, "label"] == 0).all()


# model shape tests
class TestTwoTowerModel:
    def test_forward_output_shapes(self):
        B, S = 8, 10
        num_users, num_items, num_cats = 100, 200, 30
        emb_dim = 64

        model = TwoTowerModel(
            num_users=num_users, num_items=num_items, num_categories=num_cats,
            embedding_dim=emb_dim, user_mlp_dims=[128, emb_dim], item_mlp_dims=[emb_dim],
        )
        user_id = torch.randint(0, num_users, (B,))
        history = torch.randint(0, num_items, (B, S))
        item_id = torch.randint(0, num_items, (B,))
        cat_id = torch.randint(0, num_cats, (B,))

        user_emb, item_emb = model(user_id, history, item_id, cat_id)

        assert user_emb.shape == (B, emb_dim)
        assert item_emb.shape == (B, emb_dim)

    def test_output_is_l2_normalized(self):
        B, S = 4, 5
        model = TwoTowerModel(num_users=50, num_items=100, num_categories=10,
                               embedding_dim=32, user_mlp_dims=[64, 32], item_mlp_dims=[32])
        user_emb, item_emb = model(
            torch.randint(0, 50, (B,)),
            torch.randint(0, 100, (B, S)),
            torch.randint(0, 100, (B,)),
            torch.randint(0, 10, (B,)),
        )
        norms = torch.norm(user_emb, dim=1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5)


class TestMLPRanker:
    def test_forward_output_shape(self):
        B = 16
        sparse_dims = {"user_id": 100, "item_id": 200, "category_id": 30}
        dense_dim = 9

        model = MLPRanker(sparse_dims, dense_dim, embed_dim=16, mlp_dims=[64, 32], dropout=0.1)
        model.eval()

        sparse = {
            "user_id": torch.randint(0, 100, (B,)),
            "item_id": torch.randint(0, 200, (B,)),
            "category_id": torch.randint(0, 30, (B,)),
        }
        dense = torch.randn(B, dense_dim)

        with torch.no_grad():
            logits = model(sparse, dense)

        assert logits.shape == (B,)

    def test_sigmoid_output_in_0_1(self):
        B = 32
        sparse_dims = {"user_id": 50, "item_id": 100, "category_id": 20}
        dense_dim = 9

        model = MLPRanker(sparse_dims, dense_dim, embed_dim=16, mlp_dims=[64, 32], dropout=0.0)
        model.eval()

        sparse = {
            "user_id": torch.randint(0, 50, (B,)),
            "item_id": torch.randint(0, 100, (B,)),
            "category_id": torch.randint(0, 20, (B,)),
        }
        dense = torch.randn(B, dense_dim)

        with torch.no_grad():
            logits = model(sparse, dense)
            probs = torch.sigmoid(logits)

        assert (probs >= 0).all() and (probs <= 1).all()
