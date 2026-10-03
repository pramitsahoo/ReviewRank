# ReviewRank: Amazon Reviews Two-Stage Recommendation System

A two-stage recommendation system (Retrieval → Ranking) built on Amazon Reviews data.

## Architecture

**Retrieval**: Two-Tower model (InfoNCE + in-batch negatives) → FAISS IVF-PQ index, 26K items → 200 candidates in sub-ms latency.

**Ranking**: Sparse embeddings + dense user/item/cross features → shallow MLP, re-ranks 200 candidates → Top-K.

## Quick Start

```bash
# CPU setup
pip install -r requirements.txt

# GPU setup (example: CUDA 12.8)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python scripts/download_data.py
python scripts/preprocess.py
python scripts/train_retriever.py
python scripts/train_ranker.py
python scripts/evaluate.py
```

## Performance

Dataset: Amazon Reviews 2023 — Video Games (5-core filtering, 660K train / 99K val / 99K test)

| Metric | Value |
|--------|-------|
| Retrieval Recall@200 | 0.1419 |
| Ranking AUC (val / test) | 0.7083 / 0.6855 |
| E2E NDCG@20 | 0.0081 |
| E2E Hit Rate@20 | 0.0175 |

**Retrieval Ablation (Recall@200)**

| Method | Recall@200 |
|--------|------------|
| Two-Tower + FAISS | 0.1419 |
| Random | 0.0073 |
| Popularity | 0.0000 |

**FAISS Index Comparison**

| Index Type | Recall@200 | Avg Latency |
|------------|------------|-------------|
| FlatIP | 0.1437 | 0.044 ms |
| IVF256,PQ32 | 0.1419 | 0.009 ms |

## Project Structure

```
├── configs/          # base.yaml — all hyperparameters
├── scripts/          # pipeline entry points
├── src/
│   ├── data/         # preprocessing, vocab, features, dataset
│   ├── retrieval/    # two-tower model, trainer, FAISS index
│   ├── ranking/      # MLP ranker, trainer, predictor
│   ├── eval/         # metrics, end-to-end evaluation, ablation
│   └── api/          # FastAPI serving
└── tests/            # unit tests
```

## Serving & Export

```bash
python scripts/export_onnx.py                                   # ONNX export + benchmark
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000    # start API server
# POST /recommend {"user_id": "...", "top_k": 20}
```
