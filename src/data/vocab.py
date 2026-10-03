import json
from pathlib import Path


# vocab for user_id, item_id, category
def build_vocab(df):
    vocab: dict[str, dict[str, int]] = {}
    for col in ("user_id", "item_id", "category"):
        unique_vals = sorted(df[col].dropna().unique(), key=str)
        vocab[col] = {str(v): i + 1 for i, v in enumerate(unique_vals)}
    return vocab


def save_vocab(vocab, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
