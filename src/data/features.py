import pandas as pd


def _infer_timestamp_unit(series: pd.Series) -> str:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return "s"

    max_abs = numeric.abs().max()
    if max_abs >= 1e17:
        return "ns"
    if max_abs >= 1e14:
        return "us"
    if max_abs >= 1e11:
        return "ms"
    return "s"


def _count_active_days(series: pd.Series) -> int:
    if series.dtype == "object":
        return series.nunique()

    unit = _infer_timestamp_unit(series)
    timestamps = pd.to_datetime(series, unit=unit, errors="coerce")
    return timestamps.dt.date.nunique()


def build_user_features(train_df):
    grouped = train_df.groupby("user_id")
    user_features = pd.DataFrame({
        "num_interactions": grouped["rating"].count(),
        "avg_rating": grouped["rating"].mean(),
        "std_rating": grouped["rating"].std().fillna(0.0),
        "active_days": grouped["timestamp"].apply(_count_active_days),
    })
    return user_features


def build_item_features(train_df, meta_df):
    grouped = train_df.groupby("item_id")
    item_features = pd.DataFrame({
        "num_ratings": grouped["rating"].count(),
        "avg_rating": grouped["rating"].mean(),
    })

    meta_sub = meta_df[["item_id", "price", "category"]].drop_duplicates("item_id").set_index("item_id")
    item_features = item_features.join(meta_sub, how="left")

    price = pd.to_numeric(item_features["price"], errors="coerce").fillna(0.0)
    p_min, p_max = price.min(), price.max()
    item_features["price"] = (price - p_min) / (p_max - p_min + 1e-8) # min-max normalization
    item_features["category"] = item_features["category"].fillna("unknown")

    return item_features
