import numpy as np


def recall_at_k(actual, predicted, k):
    actual_set = set(actual)
    if not actual_set:
        return 0.0
    topk = predicted[:k]
    return len(actual_set & set(topk)) / len(actual_set)


def hit_rate_at_k(actual, predicted, k):

    actual_set = set(actual)
    topk = predicted[:k]
    return 1.0 if actual_set & set(topk) else 0.0


# normalized discounted cumulative gain
def ndcg_at_k(actual, predicted, k):
    actual_set = set(actual)
    if not actual_set:
        return 0.0

    topk = predicted[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(topk) if item in actual_set)

    # ideal DCG: all relevant items at the top
    ideal_hits = min(len(actual_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


# mean reciprocal rank
def mrr(actual, predicted):
    actual_set = set(actual)
    for i, item in enumerate(predicted):
        if item in actual_set:
            return 1.0 / (i + 1)
    return 0.0
