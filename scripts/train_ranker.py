import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import RankingDataset
from src.data.features import build_user_features, build_item_features
from src.ranking.ranker import MLPRanker
from src.ranking.trainer import RankerTrainer
from src.utils import load_config, set_seed, get_device, load_splits, vocab_sizes


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    processed_dir = Path(cfg["data"]["processed_dir"])
    artifacts_dir = Path(cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, vocab = load_splits(processed_dir)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    meta_path = processed_dir / "meta.parquet"
    meta_df = pd.read_parquet(meta_path) if meta_path.exists() else None

    user_features = build_user_features(train_df)
    item_features = build_item_features(train_df, meta_df) if meta_df is not None else None

    train_dataset = RankingDataset(train_df, vocab, user_features, item_features, meta_df=meta_df, train_df=train_df)
    val_dataset = RankingDataset(val_df, vocab, user_features, item_features, meta_df=meta_df, train_df=train_df)

    rank_cfg = cfg["ranking"]
    train_loader = DataLoader(train_dataset, batch_size=rank_cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=rank_cfg["batch_size"], shuffle=False, num_workers=0)

    num_users, num_items, num_categories = vocab_sizes(vocab)

    sparse_feature_dims = {
        "user_id": num_users,
        "item_id": num_items,
        "category_id": num_categories,
    }
    dense_feature_dim = 9  # 4 user + 3 item + 2 cross

    model = MLPRanker(
        sparse_feature_dims=sparse_feature_dims,
        dense_feature_dim=dense_feature_dim,
        embed_dim=rank_cfg["embedding_dim"],
        mlp_dims=rank_cfg["mlp_dims"],
        dropout=rank_cfg["dropout"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=rank_cfg["learning_rate"])

    trainer = RankerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        artifacts_dir=str(artifacts_dir),
        epochs=rank_cfg["epochs"],
        early_stop_patience=rank_cfg["early_stop_patience"],
    )
    trainer.train()

    # final evaluation on test set
    ckpt = torch.load(artifacts_dir / "ranker_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    test_dataset = RankingDataset(test_df, vocab, user_features, item_features, meta_df=meta_df, train_df=train_df)
    test_loader = DataLoader(test_dataset, batch_size=rank_cfg["batch_size"], shuffle=False, num_workers=0)

    test_auc, test_logloss = trainer.evaluate(test_loader, desc="Test")

    print(f"\n── Ranking Test Evaluation ──")
    print(f"  AUC:     {test_auc:.4f}")
    print(f"  LogLoss: {test_logloss:.4f}")


if __name__ == "__main__":
    main()
