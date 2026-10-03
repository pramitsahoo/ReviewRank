import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm


class RankerTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, artifacts_dir="artifacts",
                 epochs=15, early_stop_patience=3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self):
        best_auc = 0.0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss = self._train_epoch(epoch)
            val_auc, val_logloss = self.evaluate(self.val_loader, desc="Val")

            elapsed = time.time() - t0
            print(f"Epoch {epoch}/{self.epochs}  "
                  f"train_loss={train_loss:.4f}  val_AUC={val_auc:.4f}  "
                  f"val_logloss={val_logloss:.4f}  time={elapsed:.1f}s")

            if val_auc > best_auc or epoch == 1:
                best_auc = val_auc
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(epoch, val_auc, val_logloss)
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    print(f"Early stopping at epoch {epoch} (patience={self.early_stop_patience})")
                    break

        print(f"Best epoch: {best_epoch}, best val_AUC: {best_auc:.4f}")
        return {"best_epoch": best_epoch, "best_val_auc": best_auc}

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}/{self.epochs}", leave=False)
        for sparse, dense, label in pbar:
            sparse = {k: v.to(self.device) for k, v in sparse.items()}
            dense = dense.to(self.device)
            label = label.to(self.device)

            pred = self.model(sparse, dense)
            loss = self.criterion(pred, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader, desc="Eval"):
        self.model.eval()
        all_preds = []
        all_labels = []

        for sparse, dense, label in tqdm(loader, desc=desc, leave=False):
            sparse = {k: v.to(self.device) for k, v in sparse.items()}
            dense = dense.to(self.device)

            logits = self.model(sparse, dense)
            pred = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(pred)
            all_labels.append(label.numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        if len(np.unique(all_labels)) < 2:
            return 0.0, float("inf")
        auc = roc_auc_score(all_labels, all_preds)
        logloss = log_loss(all_labels, all_preds)
        return auc, logloss

    def _save_checkpoint(self, epoch, val_auc, val_logloss):
        path = self.artifacts_dir / "ranker_best.pt"
        torch.save({
            "epoch": epoch,
            "val_auc": val_auc,
            "val_logloss": val_logloss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"  Saved checkpoint → {path}")
