import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import RetrievalDataset, RankingDataset
from src.data.features import build_user_features, build_item_features
from src.retrieval.two_tower import TwoTowerModel
from src.ranking.ranker import MLPRanker
from src.retrieval.index import FAISSIndex
from src.ranking.predictor import RankingPredictor
from src.eval.pipeline import EndToEndEvaluator
from src.eval.ablation import ranking_ablation, retrieval_ablation, feature_ablation
from src.utils import (load_config, set_seed, get_device, load_splits,
                       vocab_sizes, build_item_category_arrays)


def print_table(title, results):
    print(f"\n{'═' * 50}")
    print(f"  {title}")
    print(f"{'═' * 50}")
    for name, value in results.items():
        if isinstance(value, float):
            print(f"  {name:<30} {value:.4f}")
        else:
            print(f"  {name:<30} {value}")
    print()


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    processed_dir = Path(cfg["data"]["processed_dir"])
    artifacts_dir = Path(cfg["artifacts_dir"])

    # load data
    train_df, val_df, test_df, vocab = load_splits(processed_dir)
    num_users, num_items, num_categories = vocab_sizes(vocab)

    meta_path = processed_dir / "meta.parquet"
    meta_df = pd.read_parquet(meta_path) if meta_path.exists() else None

    user_features = build_user_features(train_df)
    item_features = build_item_features(train_df, meta_df) if meta_df is not None else None


    # load retrieval model + FAISS index
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
    retrieval_model.to(device).eval()

    idx_cfg = cfg["index"]
    all_item_ids, all_category_ids = build_item_category_arrays(train_df, vocab)

    faiss_index = FAISSIndex(
        embedding_dim=ret_cfg["embedding_dim"],
        index_type=idx_cfg["index_type"],
        nprobe=idx_cfg["nprobe"],
    )
    index_path = artifacts_dir / "faiss.index"
    if index_path.exists():
        faiss_index.load(str(index_path))
    else:
        faiss_index.build(
            item_tower=retrieval_model.item_tower,
            all_item_ids=all_item_ids,
            all_category_ids=all_category_ids,
            device=device,
            save_path=str(index_path),
        )

    # load ranking model
    rank_cfg = cfg["ranking"]
    sparse_feature_dims = {
        "user_id": num_users,
        "item_id": num_items,
        "category_id": num_categories,
    }
    dense_feature_dim = 9

    ranker_model = MLPRanker(
        sparse_feature_dims=sparse_feature_dims,
        dense_feature_dim=dense_feature_dim,
        embed_dim=rank_cfg["embedding_dim"],
        mlp_dims=rank_cfg["mlp_dims"],
        dropout=rank_cfg["dropout"],
    )
    ckpt = torch.load(artifacts_dir / "ranker_best.pt", map_location=device, weights_only=True)
    ranker_model.load_state_dict(ckpt["model_state_dict"])
    ranker_model.to(device).eval()

    ranker_predictor = RankingPredictor(
        ranker_model, vocab, user_features, item_features, device,
        meta_df=meta_df, train_df=train_df,
    )

    # dataLoaders
    max_hist = cfg["data"]["max_history_length"]
    test_retrieval_ds = RetrievalDataset(test_df, vocab, max_hist, history_df=train_df)
    test_retrieval_loader = DataLoader(test_retrieval_ds, batch_size=ret_cfg["batch_size"],
                                       shuffle=False, num_workers=0)

    train_ranking_ds = RankingDataset(train_df, vocab, user_features, item_features, meta_df=meta_df, train_df=train_df)
    val_ranking_ds = RankingDataset(val_df, vocab, user_features, item_features, meta_df=meta_df, train_df=train_df)
    train_ranking_loader = DataLoader(train_ranking_ds, batch_size=rank_cfg["batch_size"],
                                      shuffle=True, num_workers=0)
    val_ranking_loader = DataLoader(val_ranking_ds, batch_size=rank_cfg["batch_size"],
                                    shuffle=False, num_workers=0)

    all_results = {}

    # 1. end-to-end evaluation
    print("\nRunning end-to-end evaluation...")
    ranking_k_list = cfg["evaluation"]["ranking_k"]
    e2e = EndToEndEvaluator(
        retrieval_model=retrieval_model,
        ranker_predictor=ranker_predictor,
        faiss_index=faiss_index,
        vocab=vocab,
        device=device,
        retrieval_top_k=idx_cfg["top_k"],
        ranking_k_list=ranking_k_list,
    )
    e2e_results = e2e.evaluate(test_retrieval_loader)
    print_table("End-to-End Metrics", e2e_results)
    all_results["end_to_end"] = e2e_results

    # 2. retrieval ablation
    print("Running retrieval ablation...")
    ret_abl = retrieval_ablation(
        retrieval_model, faiss_index, test_retrieval_loader, device,
        all_item_ids, train_df, top_k=idx_cfg["top_k"], k=200,
    )
    print_table("Retrieval Ablation (Recall@200)", ret_abl)
    all_results["retrieval_ablation"] = ret_abl

    # 3. ranking model ablation
    print("Running ranking model ablation...")
    rank_abl = ranking_ablation(
        sparse_feature_dims=sparse_feature_dims,
        dense_feature_dim=dense_feature_dim,
        embed_dim=rank_cfg["embedding_dim"],
        mlp_dims=rank_cfg["mlp_dims"],
        dropout=rank_cfg["dropout"],
        train_loader=train_ranking_loader,
        val_loader=val_ranking_loader,
        device=device,
        lr=rank_cfg["learning_rate"],
        epochs=rank_cfg["epochs"],
        patience=rank_cfg["early_stop_patience"],
    )
    print_table("Ranking Model Ablation (Val AUC)", rank_abl)
    all_results["ranking_model_ablation"] = rank_abl

    # 4. feature ablation
    print("Running feature ablation...")
    feat_abl = feature_ablation(ranker_model, val_ranking_loader, device)
    print_table("Feature Ablation (Val AUC)", feat_abl)
    all_results["feature_ablation"] = feat_abl

    # results
    output_path = artifacts_dir / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
