import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from src.utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    raw_dir = PROJECT_ROOT / data_cfg["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = data_cfg["dataset_name"]
    hf_dataset = data_cfg["hf_dataset"]

    reviews_config = f"raw_review_{dataset_name}"
    meta_config = f"raw_meta_{dataset_name}"

    print(f"Downloading {dataset_name} reviews to {raw_dir} ...")

    reviews = load_dataset(
        hf_dataset,
        reviews_config,
        split="full",
        trust_remote_code=True,
    )
    reviews_path = raw_dir / "reviews.parquet"
    reviews.to_parquet(str(reviews_path))
    print(f"Saved reviews to {reviews_path} ({len(reviews)} rows)")

    meta = load_dataset(
        hf_dataset,
        meta_config,
        split="full",
        trust_remote_code=True,
    )
    meta_path = raw_dir / "meta.parquet"
    meta.to_parquet(str(meta_path))
    print(f"Saved metadata to {meta_path} ({len(meta)} rows)")


if __name__ == "__main__":
    main()
