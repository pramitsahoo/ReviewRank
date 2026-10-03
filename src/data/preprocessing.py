import pandas as pd
from pathlib import Path


def load_and_filter(raw_dir, k_core):
    raw_path = Path(raw_dir)
    reviews_path = raw_path / "reviews.parquet"
    if not reviews_path.exists():
        raise FileNotFoundError(f"reviews.parquet not found in {raw_dir}")
    df = pd.read_parquet(reviews_path)

    # rename raw Amazon columns to internal names
    rename_map = {}
    if "parent_asin" in df.columns and "item_id" not in df.columns:
        rename_map["parent_asin"] = "item_id"
    if "main_category" in df.columns and "category" not in df.columns:
        rename_map["main_category"] = "category"
    if rename_map:
        df = df.rename(columns=rename_map)

    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= k_core].index)]
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= k_core].index)]

    df = df.reset_index(drop=True)
    return df


def binarize_labels(df, threshold):
    df = df.copy()
    df["label"] = (df["rating"] >= threshold).astype(int)
    return df


def leave_last_out_split(df):
    # Split per user based on timestamp.
    # For each user: last → test, second-to-last → val, rest → train.
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    train_rows, val_rows, test_rows = [], [], []

    for _, group in df.groupby("user_id"):
        if len(group) < 3:
            train_rows.append(group) # less than 3 interactions → all to train
            continue
        train_rows.append(group.iloc[:-2])
        val_rows.append(group.iloc[[-2]])
        test_rows.append(group.iloc[[-1]])

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=df.columns)

    return train_df, val_df, test_df
