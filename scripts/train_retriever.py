import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import RetrievalDataset
from src.retrieval.two_tower import TwoTowerModel
from src.retrieval.trainer import RetrieverTrainer
from src.retrieval.index import FAISSIndex
from src.utils import load_config, set_seed, get_device, load_splits, vocab_sizes, build_item_category_arrays


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

    max_hist = cfg["data"]["max_history_length"]
    train_dataset = RetrievalDataset(train_df, vocab, max_hist)
    val_dataset = RetrievalDataset(val_df, vocab, max_hist, history_df=train_df)
    test_dataset = RetrievalDataset(test_df, vocab, max_hist, history_df=train_df)

    ret_cfg = cfg["retrieval"]
    train_loader = DataLoader(train_dataset, batch_size=ret_cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=ret_cfg["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=ret_cfg["batch_size"], shuffle=False, num_workers=0)

    num_users, num_items, num_categories = vocab_sizes(vocab)

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        num_categories=num_categories,
        embedding_dim=ret_cfg["embedding_dim"],
        user_mlp_dims=ret_cfg["mlp_dims"],
        item_mlp_dims=[ret_cfg["embedding_dim"]],
        temperature=ret_cfg["temperature"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=ret_cfg["learning_rate"])

    trainer = RetrieverTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        artifacts_dir=str(artifacts_dir),
        epochs=ret_cfg["epochs"],
        early_stop_patience=ret_cfg["early_stop_patience"],
        temperature=ret_cfg["temperature"],
    )
    trainer.train()

    # load best checkpoint
    ckpt = torch.load(artifacts_dir / "retriever_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"\nLoaded best checkpoint from epoch {ckpt['epoch']}")

    # build FAISS index
    all_item_ids, all_category_ids = build_item_category_arrays(train_df, vocab)

    idx_cfg = cfg["index"]
    faiss_index = FAISSIndex(
        embedding_dim=ret_cfg["embedding_dim"],
        index_type=idx_cfg["index_type"],
        nprobe=idx_cfg["nprobe"],
    )
    faiss_index.build(
        item_tower=model.item_tower,
        all_item_ids=all_item_ids,
        all_category_ids=all_category_ids,
        device=device,
        save_path=str(artifacts_dir / "faiss.index"),
    )

    # evaluate Recall@K
    eval_k = cfg["evaluation"]["retrieval_k"]
    recall_results = faiss_index.evaluate(model.user_tower, test_loader, device, k_list=eval_k)

    print("\n── Retrieval Evaluation ──")
    for metric, value in recall_results.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
