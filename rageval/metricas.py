"""Métricas de recuperação com relevância binária."""
import math


def recall_at_k(ranking: list[str], relevantes: set[str], k: int) -> float:
    if not relevantes:
        return 0.0
    return len(set(ranking[:k]) & relevantes) / len(relevantes)


def hit_at_k(ranking: list[str], relevantes: set[str], k: int) -> float:
    return 1.0 if set(ranking[:k]) & relevantes else 0.0


def mrr(ranking: list[str], relevantes: set[str]) -> float:
    """Inverso da posição do primeiro relevante (0 se nenhum aparece no ranking)."""
    for pos, id_ in enumerate(ranking, start=1):
        if id_ in relevantes:
            return 1.0 / pos
    return 0.0


def ndcg_at_k(ranking: list[str], relevantes: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(pos + 1) for pos, id_ in enumerate(ranking[:k], start=1) if id_ in relevantes)
    ideal = sum(1.0 / math.log2(pos + 1) for pos in range(1, min(len(relevantes), k) + 1))
    return dcg / ideal if ideal else 0.0
