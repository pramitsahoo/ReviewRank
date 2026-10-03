import numpy as np
import torch
from tqdm import tqdm
from src.eval.metrics import recall_at_k, hit_rate_at_k, ndcg_at_k, mrr


class EndToEndEvaluator:
    def __init__(self, retrieval_model, ranker_predictor, faiss_index, vocab, device,
                 retrieval_top_k=200, ranking_k_list=[10, 20]):
        self.retrieval_model = retrieval_model
        self.ranker = ranker_predictor
        self.faiss_index = faiss_index
        self.vocab = vocab
        self.device = device
        self.retrieval_top_k = retrieval_top_k
        self.ranking_k_list = ranking_k_list

        self.id_to_raw_item = {v: k for k, v in vocab["item_id"].items()}
        self.id_to_raw_user = {v: k for k, v in vocab["user_id"].items()}

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.retrieval_model.eval()
        user_tower = self.retrieval_model.user_tower

        all_metrics = {f"{m}@{k}": [] for k in self.ranking_k_list
                       for m in ("ndcg", "hit_rate", "recall")}
        all_metrics["mrr"] = []

        max_k = max(self.ranking_k_list)

        for user_id, user_history, pos_item_id, category_id in tqdm(test_loader, desc="E2E eval"):
            user_id_enc = user_id.to(self.device)
            user_history_dev = user_history.to(self.device)

            user_emb = user_tower(user_id_enc, user_history_dev).cpu().numpy().astype(np.float32)
            _, retrieved_ids = self.faiss_index.search(user_emb, self.retrieval_top_k)

            pos_items = pos_item_id.numpy()
            B = len(pos_items)

            # convert to raw IDs for batch ranking
            user_ids_raw = [self.id_to_raw_user.get(int(user_id[i]), str(int(user_id[i])))
                            for i in range(B)]
            candidates_raw = [[self.id_to_raw_item.get(c, str(c)) for c in retrieved_ids[i].tolist()]
                              for i in range(B)]

            # batch ranking: one forward pass for entire batch
            scores = self.ranker.predict_batch(user_ids_raw, candidates_raw)  # (B, C)

            for i in range(B):
                gt = [int(pos_items[i])]
                top_indices = np.argsort(scores[i])[::-1][:max_k]
                ranked_encoded = [self.vocab["item_id"].get(str(candidates_raw[i][j]), 0)
                                  for j in top_indices]

                for k in self.ranking_k_list:
                    all_metrics[f"ndcg@{k}"].append(ndcg_at_k(gt, ranked_encoded, k))
                    all_metrics[f"hit_rate@{k}"].append(hit_rate_at_k(gt, ranked_encoded, k))
                    all_metrics[f"recall@{k}"].append(recall_at_k(gt, ranked_encoded, k))
                all_metrics["mrr"].append(mrr(gt, ranked_encoded))

        return {name: float(np.mean(vals)) for name, vals in all_metrics.items() if vals}
