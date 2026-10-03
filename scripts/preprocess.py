import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import load_and_filter, binarize_labels, leave_last_out_split
from src.data.vocab import build_vocab, save_vocab
from src.data.features import build_user_features, build_item_features
from src.utils import load_config


def main():
    cfg = load_config()

    data_cfg = cfg["data"]
    raw_dir = str(PROJECT_ROOT / data_cfg["raw_dir"])
    processed_dir = PROJECT_ROOT / data_cfg["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    meta_df = None

    # load and k-core filter
    print(f"Loading data from {raw_dir} ...")
    df = load_and_filter(raw_dir, k_core=data_cfg["k_core"])
    print(f"After {data_cfg['k_core']}-core filtering: {len(df)} interactions, "
          f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")

    meta_path = Path(raw_dir) / "meta.parquet"
    if meta_path.exists():
        meta_df = pd.read_parquet(meta_path)
        meta_rename = {}
        if "parent_asin" in meta_df.columns and "item_id" not in meta_df.columns:
            meta_rename["parent_asin"] = "item_id"
        if "main_category" in meta_df.columns and "category" not in meta_df.columns:
            meta_rename["main_category"] = "category"
        if meta_rename:
            meta_df = meta_df.rename(columns=meta_rename)

        if "category" not in df.columns:
            category_lookup = meta_df[["item_id", "category"]].drop_duplicates("item_id")
            df = df.merge(category_lookup, on="item_id", how="left")

    if "category" not in df.columns:
        df["category"] = "unknown"
    else:
        df["category"] = df["category"].fillna("unknown")

    # binarize labels
    df = binarize_labels(df, threshold=data_cfg["positive_threshold"])
    pos_ratio = df["label"].mean()
    print(f"Positive label ratio: {pos_ratio:.2%}")

    # leave-last-out split
    train_df, val_df, test_df = leave_last_out_split(df)
    print(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # build and save vocab
    vocab = build_vocab(df)
    vocab_path = str(processed_dir / "vocab.json")
    save_vocab(vocab, vocab_path)
    print(f"Vocab saved to {vocab_path}")
    for col, mapping in vocab.items():
        print(f"  {col}: {len(mapping)} unique values")

    # build features
    user_features = build_user_features(train_df)
    user_features.to_parquet(processed_dir / "user_features.parquet")
    print(f"User features: {len(user_features)} users")

    if meta_df is not None:
        meta_df.to_parquet(processed_dir / "meta.parquet", index=False)
        item_features = build_item_features(train_df, meta_df)
    else:
        print("Warning: meta.parquet not found, building item features without metadata")
        grouped = train_df.groupby("item_id")
        item_features = pd.DataFrame({
            "num_ratings": grouped["rating"].count(),
            "avg_rating": grouped["rating"].mean(),
            "price": 0.0,
            "category": "unknown",
        })
    item_features.to_parquet(processed_dir / "item_features.parquet")
    print(f"Item features: {len(item_features)} items")

    # save splits
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)
    print(f"Splits saved to {processed_dir}")

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
