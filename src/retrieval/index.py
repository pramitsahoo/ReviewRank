from pathlib import Path
import faiss
import numpy as np
import torch
from tqdm import tqdm


class FAISSIndex:
    def __init__(self, embedding_dim=128, index_type="FlatIP", nprobe=32):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nprobe = nprobe
        self.index = None
        self.item_ids = None

    @torch.no_grad()
    def build(self, item_tower, all_item_ids, all_category_ids,
              device, batch_size=4096, save_path=None):
        item_tower.eval()
        embeddings = []

        for start in tqdm(range(0, len(all_item_ids), batch_size), desc="Encoding items", leave=False):
            end = start + batch_size
            iids = torch.tensor(all_item_ids[start:end], dtype=torch.long, device=device)
            cids = torch.tensor(all_category_ids[start:end], dtype=torch.long, device=device)
            emb = item_tower(iids, cids).cpu().numpy()
            embeddings.append(emb)

        embeddings = np.vstack(embeddings).astype(np.float32)
        self.item_ids = all_item_ids.copy()

        if self.index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            self.index = faiss.index_factory(self.embedding_dim, self.index_type, faiss.METRIC_INNER_PRODUCT)
            if not self.index.is_trained:
                self.index.train(embeddings)

        self.index.add(embeddings)

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

        print(f"Built FAISS index: type={self.index_type}, n_items={self.index.ntotal}")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, save_path)
            np.save(save_path + ".item_ids.npy", self.item_ids)
            print(f"Saved index → {save_path}")

    def load(self, path):
        self.index = faiss.read_index(path)
        self.item_ids = np.load(path + ".item_ids.npy")
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe
        print(f"Loaded FAISS index: n_items={self.index.ntotal}")

    def search(self, user_vectors, top_k):
        distances, faiss_indices = self.index.search(user_vectors.astype(np.float32), top_k)
        item_indices = self.item_ids[faiss_indices]
        return distances, item_indices

    @torch.no_grad()
    def evaluate(self, user_tower, test_loader, device, k_list=[50, 100, 200]):
        user_tower.eval()
        max_k = max(k_list)

        all_user_embeds = []
        all_positive_items = []

        for user_id, user_history, item_id, category_id in tqdm(test_loader, desc="FAISS eval", leave=False):
            user_id = user_id.to(device)
            user_history = user_history.to(device)
            user_emb = user_tower(user_id, user_history).cpu().numpy()
            all_user_embeds.append(user_emb)
            all_positive_items.append(item_id.numpy())

        all_user_embeds = np.vstack(all_user_embeds).astype(np.float32)
        all_positive_items = np.concatenate(all_positive_items)

        _, retrieved_items = self.search(all_user_embeds, max_k)

        results = {}
        for k in k_list:
            topk = retrieved_items[:, :k]
            hits = (topk == all_positive_items[:, None]).any(axis=1).mean()
            results[f"recall@{k}"] = float(hits)

        return results
