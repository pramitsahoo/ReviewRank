import time
from pathlib import Path
import torch
from tqdm import tqdm


class RetrieverTrainer:
    def __init__(self, model, train_loader, val_loader,
                 optimizer, device, artifacts_dir="artifacts", epochs=20,
                 early_stop_patience=3, temperature=0.05):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.temperature = temperature

    def train(self):
        best_recall = 0.0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss = self._train_epoch(epoch)
            val_recall = self._validate()

            elapsed = time.time() - t0
            print(f"Epoch {epoch}/{self.epochs}  "
                  f"train_loss={train_loss:.4f}  val_inbatch_recall={val_recall:.4f}  "
                  f"time={elapsed:.1f}s")

            if val_recall > best_recall or epoch == 1:
                best_recall = val_recall
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(epoch, val_recall)
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    print(f"Early stopping at epoch {epoch} (patience={self.early_stop_patience})")
                    break

        print(f"Best epoch: {best_epoch}, best val_inbatch_recall: {best_recall:.4f}")
        return {"best_epoch": best_epoch, "best_val_inbatch_recall": best_recall}

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}/{self.epochs}", leave=False)
        for user_id, user_history, item_id, category_id in pbar:
            user_id = user_id.to(self.device)
            user_history = user_history.to(self.device)
            item_id = item_id.to(self.device)
            category_id = category_id.to(self.device)

            user_embeds, item_embeds = self.model(user_id, user_history, item_id, category_id)
            loss = self.model.compute_loss(user_embeds, item_embeds, self.temperature)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, top_k=200):
        # in-batch recall proxy (not full-catalog); real eval via FAISS in train_retriever.py
        self.model.eval()
        hits = 0
        total = 0

        for user_id, user_history, item_id, category_id in tqdm(self.val_loader, desc="Val", leave=False):
            user_id = user_id.to(self.device)
            user_history = user_history.to(self.device)
            item_id = item_id.to(self.device)
            category_id = category_id.to(self.device)

            user_embeds, item_embeds = self.model(user_id, user_history, item_id, category_id)
            sim = user_embeds @ item_embeds.T
            B = sim.size(0)
            k = min(top_k, B)
            _, topk_indices = sim.topk(k, dim=1)
            labels = torch.arange(B, device=sim.device).unsqueeze(1)
            hits += (topk_indices == labels).any(dim=1).sum().item()
            total += B

        return hits / max(total, 1)

    def _save_checkpoint(self, epoch, val_recall):
        path = self.artifacts_dir / "retriever_best.pt"
        torch.save({
            "epoch": epoch,
            "val_inbatch_recall": val_recall,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"  Saved checkpoint → {path}")
