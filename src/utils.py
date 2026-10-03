from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from src.data.vocab import load_vocab

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_name = "base.yaml"):
    path = PROJECT_ROOT / "configs" / config_name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load train/val/test parquet files and vocab
def load_splits(processed_dir):
    p = Path(processed_dir)
    train_df = pd.read_parquet(p / "train.parquet")
    val_df = pd.read_parquet(p / "val.parquet")
    test_df = pd.read_parquet(p / "test.parquet")
    vocab = load_vocab(str(p / "vocab.json"))
    return train_df, val_df, test_df, vocab


def vocab_sizes(vocab):
    return (
        len(vocab["user_id"]) + 1,
        len(vocab["item_id"]) + 1,
        len(vocab["category"]) + 1,
    )


# build aligned arrays of vocab-encoded item IDs and their category IDs
def build_item_category_arrays(train_df, vocab):
    all_item_ids = np.array(sorted(vocab["item_id"].values()), dtype=np.int64)
    item_to_cat = {}
    for _, row in train_df.drop_duplicates("item_id").iterrows():
        iid = vocab["item_id"].get(str(row["item_id"]), 0)
        cid = vocab["category"].get(str(row["category"]), 0)
        item_to_cat[iid] = cid
    all_category_ids = np.array([item_to_cat.get(iid, 0) for iid in all_item_ids], dtype=np.int64)
    return all_item_ids, all_category_ids


def build_user_cat_stats(train_df, meta_df=None):
    if "category" in train_df.columns:
        src = train_df
    elif meta_df is not None:
        item_to_cat = meta_df[["item_id", "category"]].drop_duplicates("item_id")
        src = train_df.merge(item_to_cat, on="item_id", how="left")
    else:
        return {}
    stats = {}
    for (uid, cat), grp in src.groupby(["user_id", "category"]):
        stats[(uid, cat)] = float(grp["rating"].mean())
    return stats
