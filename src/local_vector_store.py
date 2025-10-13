import json
import os
from math import sqrt
from typing import Any, Dict, List, Iterable, Optional


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _l2(a: List[float]) -> float:
    return sqrt(sum(x * x for x in a))


def cosine(a: List[float], b: List[float]) -> float:
    da = _l2(a)
    db = _l2(b)
    if da == 0.0 or db == 0.0:
        return 0.0
    return _dot(a, b) / (da * db)


def save_points_jsonl(path: str, points: Iterable[Dict[str, Any]]) -> None:
    """
    Overwrite path with one JSON object per line.
    Each point should be a dict with at least: id, vector (list[float]), payload (dict)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in points:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def load_points_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # skip malformed lines
                continue
    return out


def cosine_search(points: List[Dict[str, Any]], query_vector: List[float], limit: int = 30) -> List[Dict[str, Any]]:
    """
    Return a list of hits in descending cosine score.
    Hit schema mirrors Qdrant enough for UI:
      { "id": id, "score": float, "payload": dict, "vector": list[float] }
    """
    scored: List[Dict[str, Any]] = []
    for p in points:
        vec = p.get("vector") or []
        if not isinstance(vec, list) or not vec:
            continue
        score = cosine(query_vector, vec)
        scored.append({
            "id": p.get("id"),
            "score": float(score),
            "payload": p.get("payload") or {},
            "vector": vec,
        })
    scored.sort(key=lambda h: h["score"], reverse=True)
    return scored[: max(limit, 0)]