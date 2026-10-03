import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

from src.ranking.ranker import MLPRanker
from src.utils import load_config, vocab_sizes, load_splits


# wrapper to adapt MLPRanker for ONNX export (dict input → separate tensors)
class RankerONNXWrapper(nn.Module):
    def __init__(self, ranker: MLPRanker):
        super().__init__()
        self.ranker = ranker

    def forward(self, user_id, item_id, category_id, dense_features):
        sparse = {"user_id": user_id, "item_id": item_id, "category_id": category_id}
        return self.ranker(sparse, dense_features)


def benchmark(fn, warmup=10, runs=100):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t0) / runs * 1000


def main():
    cfg = load_config()
    device = torch.device("cpu")

    processed_dir = Path(cfg["data"]["processed_dir"])
    artifacts_dir = Path(cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _, _, _, vocab = load_splits(processed_dir)
    num_users, num_items, num_categories = vocab_sizes(vocab)

    rank_cfg = cfg["ranking"]
    sparse_feature_dims = {
        "user_id": num_users,
        "item_id": num_items,
        "category_id": num_categories,
    }
    dense_feature_dim = 9

    # load checkpoint
    model = MLPRanker(
        sparse_feature_dims=sparse_feature_dims,
        dense_feature_dim=dense_feature_dim,
        embed_dim=rank_cfg["embedding_dim"],
        mlp_dims=rank_cfg["mlp_dims"],
        dropout=rank_cfg["dropout"],
    )
    ckpt = torch.load(artifacts_dir / "ranker_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapper = RankerONNXWrapper(model)
    wrapper.eval()

    # export to ONNX
    onnx_path = str(artifacts_dir / "ranker.onnx")
    dummy_user = torch.tensor([1], dtype=torch.long)
    dummy_item = torch.tensor([1], dtype=torch.long)
    dummy_cat = torch.tensor([1], dtype=torch.long)
    dummy_dense = torch.randn(1, dense_feature_dim)

    torch.onnx.export(
        wrapper,
        (dummy_user, dummy_item, dummy_cat, dummy_dense),
        onnx_path,
        input_names=["user_id", "item_id", "category_id", "dense_features"],
        output_names=["logit"],
        dynamic_axes={
            "user_id": {0: "batch_size"},
            "item_id": {0: "batch_size"},
            "category_id": {0: "batch_size"},
            "dense_features": {0: "batch_size"},
            "logit": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"Exported ONNX model → {onnx_path}")

    # validate consistency
    session = ort.InferenceSession(onnx_path)

    batch_size = 32
    test_user = torch.randint(0, num_users, (batch_size,))
    test_item = torch.randint(0, num_items, (batch_size,))
    test_cat = torch.randint(0, num_categories, (batch_size,))
    test_dense = torch.randn(batch_size, dense_feature_dim)

    with torch.no_grad():
        pt_output = wrapper(test_user, test_item, test_cat, test_dense).numpy()

    ort_output = session.run(None, {
        "user_id": test_user.numpy(),
        "item_id": test_item.numpy(),
        "category_id": test_cat.numpy(),
        "dense_features": test_dense.numpy(),
    })[0]

    max_diff = np.max(np.abs(pt_output - ort_output))
    print(f"Max absolute difference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Consistency check FAILED: max_diff={max_diff:.2e}"
    print("Consistency check PASSED (< 1e-5)\n")

    # benchmark
    batch_sizes = [1, 32, 200]

    print(f"{'Batch':<8} {'PyTorch (ms)':>14} {'ONNX RT (ms)':>14} {'Speedup':>10}")
    print("-" * 50)

    for bs in batch_sizes:
        b_user = torch.randint(0, num_users, (bs,))
        b_item = torch.randint(0, num_items, (bs,))
        b_cat = torch.randint(0, num_categories, (bs,))
        b_dense = torch.randn(bs, dense_feature_dim)

        # PyTorch
        def pt_fn():
            with torch.no_grad():
                wrapper(b_user, b_item, b_cat, b_dense)

        pt_ms = benchmark(pt_fn)

        # ONNX runtime
        ort_inputs = {
            "user_id": b_user.numpy(),
            "item_id": b_item.numpy(),
            "category_id": b_cat.numpy(),
            "dense_features": b_dense.numpy(),
        }

        def ort_fn():
            session.run(None, ort_inputs)

        ort_ms = benchmark(ort_fn)

        speedup = pt_ms / ort_ms
        print(f"{bs:<8} {pt_ms:>14.3f} {ort_ms:>14.3f} {speedup:>9.2f}x")

    print(f"\nONNX model saved to {onnx_path}")


if __name__ == "__main__":
    main()
