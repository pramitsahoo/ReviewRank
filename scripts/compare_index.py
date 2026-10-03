import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import RetrievalDataset
from src.retrieval.two_tower import TwoTowerModel
from src.retrieval.index import FAISSIndex
from src.utils import load_config, get_device, load_splits, vocab_sizes, build_item_category_arrays


# compare FLAT vs IVF-PQ index types for Recall@200 and latency
def main():
    cfg = load_config()
    device = get_device()

    processed_dir = Path(cfg["data"]["processed_dir"])
    artifacts_dir = Path(cfg["artifacts_dir"])

    train_df, _, test_df, vocab = load_splits(processed_dir)

    ret_cfg = cfg["retrieval"]
    test_dataset = RetrievalDataset(test_df, vocab, cfg["data"]["max_history_length"], history_df=train_df)
    test_loader = DataLoader(test_dataset, batch_size=ret_cfg["batch_size"], shuffle=False, num_workers=0)

    # load trained model
    num_users, num_items, num_categories = vocab_sizes(vocab)

    model = TwoTowerModel(
        num_users=num_users, num_items=num_items, num_categories=num_categories,
        embedding_dim=ret_cfg["embedding_dim"],
        user_mlp_dims=ret_cfg["mlp_dims"],
        item_mlp_dims=[ret_cfg["embedding_dim"]],
        temperature=ret_cfg["temperature"],
    )
    ckpt = torch.load(artifacts_dir / "retriever_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # prepare item embeddings data
    all_item_ids, all_category_ids = build_item_category_arrays(train_df, vocab)

    # encode all test users once
    all_user_embeds = []
    all_positive_items = []
    with torch.no_grad():
        for user_id, user_history, item_id, category_id in test_loader:
            user_emb = model.user_tower(user_id.to(device), user_history.to(device)).cpu().numpy()
            all_user_embeds.append(user_emb)
            all_positive_items.append(item_id.numpy())
    all_user_embeds = np.vstack(all_user_embeds).astype(np.float32)
    all_positive_items = np.concatenate(all_positive_items)

    # compare different FAISS index types
    idx_cfg = cfg["index"]
    index_configs = [
        ("FlatIP", "FlatIP", 0),
        ("IVF-PQ", idx_cfg["index_type"], idx_cfg["nprobe"]),
    ]
    top_k = idx_cfg["top_k"]

    print(f"Test queries: {len(all_user_embeds)}, Items in index: {len(all_item_ids)}, Top-K: {top_k}\n")
    print(f"{'Index Type':<15} {'Recall@200':>12} {'Avg Latency (ms)':>18}")
    print("-" * 48)

    for name, index_type, nprobe in index_configs:
        idx = FAISSIndex(embedding_dim=ret_cfg["embedding_dim"], index_type=index_type, nprobe=nprobe)
        idx.build(
            item_tower=model.item_tower,
            all_item_ids=all_item_ids,
            all_category_ids=all_category_ids,
            device=device,
        )

        # warm-up search
        idx.search(all_user_embeds[:10], top_k)

        # time the search
        t0 = time.perf_counter()
        _, retrieved = idx.search(all_user_embeds, top_k)
        elapsed = time.perf_counter() - t0

        # Recall@200
        k = min(top_k, 200)
        topk = retrieved[:, :k]
        recall = (topk == all_positive_items[:, None]).any(axis=1).mean()
        avg_latency_ms = elapsed / len(all_user_embeds) * 1000

        print(f"{name:<15} {recall:>12.4f} {avg_latency_ms:>15.3f} ms")

    print("\nDone.")


if __name__ == "__main__":
    main()
