"""Scalable hybrid multimodal-vector and local lexical retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import llm
from .config import settings
from .graph_store import GraphStore


def _tokens(text: str) -> set:
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9_+.*-]{2,}", lowered))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    chars = set(chinese)
    bigrams = {chinese[index:index + 2] for index in range(max(0, len(chinese) - 1))}
    return words | chars | bigrams


def _lexical_score(query: str, document: str) -> float:
    query_tokens, document_tokens = _tokens(query), _tokens(document)
    if not query_tokens or not document_tokens:
        return 0.0
    overlap = len(query_tokens & document_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(document_tokens))


def _cosine(first: List[float], second: List[float]) -> float:
    if not first or not second or len(first) != len(second):
        return 0.0
    dot = sum(x * y for x, y in zip(first, second))
    norm_first = math.sqrt(sum(x * x for x in first))
    norm_second = math.sqrt(sum(y * y for y in second))
    return dot / (norm_first * norm_second) if norm_first and norm_second else 0.0


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


class EmbeddingStore:
    """Lazy vector index with per-node hashes and a lexical safety net.

    The previous implementation embedded every node before the first answer.
    With the newly trained 1k+ node graph that meant more than one thousand
    remote requests. We now rank locally first and embed only a small candidate
    pool, while keeping unchanged node vectors across graph edits.
    """

    def __init__(self, graph: GraphStore):
        self.graph = graph
        self.vectors: Dict[str, List[float]] = {}
        self.vector_hashes: Dict[str, str] = {}
        self.mode = "local"
        self.last_error = ""
        self._cache_loaded = False
        self._cache_path: Optional[Path] = None
        self._lock = asyncio.Lock()

    def _load_cache(self) -> None:
        path = settings.emb_cache_path
        if self._cache_loaded and self._cache_path == path:
            return
        self._cache_loaded = True
        self._cache_path = path
        self.vectors = {}
        self.vector_hashes = {}
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("model") != settings.embed_model or data.get("dimension") != settings.embed_dim:
                return

            # Version 2 cache survives GraphML mtime changes because every
            # vector is tied to the exact text used to create it.
            records = data.get("records", {})
            if isinstance(records, dict):
                for node_id, record in records.items():
                    if not isinstance(record, dict):
                        continue
                    vector = record.get("vector")
                    text_hash = record.get("text_hash")
                    if isinstance(vector, list) and text_hash:
                        self.vectors[node_id] = vector
                        self.vector_hashes[node_id] = str(text_hash)
                if self.vectors:
                    self.mode = "qwen3-vl-cache"
                return

            # Backward compatibility for the former all-or-nothing cache.
            graph_mtime = settings.graphml_path.stat().st_mtime_ns if settings.graphml_path.exists() else -1
            if data.get("graph_mtime") != graph_mtime:
                return
            vectors = data.get("vectors", {})
            if isinstance(vectors, dict):
                for node_id, vector in vectors.items():
                    if node_id in self.graph.nodes and isinstance(vector, list):
                        text = self.graph.embedding_text(node_id)
                        self.vectors[node_id] = vector
                        self.vector_hashes[node_id] = _text_hash(text)
                if self.vectors:
                    self.mode = "qwen3-vl-cache"
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            self.vectors = {}
            self.vector_hashes = {}

    def _save_cache(self) -> None:
        path = settings.emb_cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        active_nodes = set(self.graph.nodes)
        records = {
            node_id: {"text_hash": self.vector_hashes[node_id], "vector": vector}
            for node_id, vector in self.vectors.items()
            if node_id in active_nodes and node_id in self.vector_hashes
        }
        payload = {
            "version": 2,
            "model": settings.embed_model,
            "dimension": settings.embed_dim,
            "records": records,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        temporary.replace(path)

    async def _ensure_candidate_vectors(self, candidate_ids: List[str]) -> None:
        self._load_cache()
        if not settings.has_key or not candidate_ids:
            return
        async with self._lock:
            texts = {node_id: self.graph.embedding_text(node_id) for node_id in candidate_ids}
            missing = [
                node_id for node_id in candidate_ids
                if node_id not in self.vectors or self.vector_hashes.get(node_id) != _text_hash(texts[node_id])
            ]
            if not missing:
                self.mode = "qwen3-vl-cache"
                return
            try:
                vectors = await llm.embed_texts([texts[node_id] for node_id in missing])
                for node_id, vector in zip(missing, vectors):
                    if vector:
                        self.vectors[node_id] = vector
                        self.vector_hashes[node_id] = _text_hash(texts[node_id])
                self._save_cache()
                self.mode = "qwen3-vl-lazy"
                self.last_error = ""
            except (llm.LLMError, OSError) as exc:
                self.last_error = str(exc)
                self.mode = "local"

    def _local_candidates(self, query: str, images: List[str], limit: int) -> Tuple[List[str], Dict[str, float]]:
        lexical_scores: Dict[str, float] = {}
        compact_query = re.sub(r"\s+", "", query).lower()
        image_request = bool(images) or any(word in query for word in ("图", "图片", "图像", "截图", "照片"))
        requested_image_numbers = {
            str(int(value))
            for value in re.findall(r"(?:图片|图像|图)\s*0*(\d+)", query, flags=re.IGNORECASE)
        }
        for node_id, record in self.graph.nodes.items():
            score = _lexical_score(query, self.graph.embedding_text(node_id))
            compact_name = re.sub(r"\s+", "", node_id).lower()
            name_position = compact_query.find(compact_name) if len(compact_name) >= 2 else -1
            if name_position >= 0:
                score += 0.65 if name_position == 0 else 0.30
            if record.get("is_image"):
                image_number = re.match(r"^图\s*0*(\d+)", node_id, flags=re.IGNORECASE)
                if image_number and str(int(image_number.group(1))) in requested_image_numbers:
                    # Explicit references such as 图53 / 图片53 / 图像53 must
                    # deterministically beat generic visual-semantic matches.
                    score += 3.0
                elif image_request:
                    score += 0.025
            lexical_scores[node_id] = score
        ranked = sorted(
            self.graph.nodes,
            key=lambda node_id: (lexical_scores[node_id], self.graph.nodes[node_id].get("degree", 0)),
            reverse=True,
        )
        return ranked[:limit], lexical_scores

    async def search(self, query: str, images: Optional[List[str]] = None, top_k: Optional[int] = None) -> List[dict]:
        images = images or []
        limit = top_k or settings.top_k
        pool_size = min(
            max(limit * 4, settings.embed_candidate_pool),
            max(1, len(self.graph.nodes)),
        )
        candidate_ids, lexical_scores = self._local_candidates(query, images, pool_size)
        await self._ensure_candidate_vectors(candidate_ids)

        query_vector: List[float] = []
        if settings.has_key and any(node_id in self.vectors for node_id in candidate_ids):
            try:
                query_vector = await llm.embed_multimodal(text=query, images=images)
                self.last_error = ""
                if self.mode == "local":
                    self.mode = "qwen3-vl-cache"
            except llm.LLMError as exc:
                self.last_error = str(exc)
                self.mode = "local"

        candidate_set = set(candidate_ids)
        scored = []
        for node_id in self.graph.nodes:
            lexical = lexical_scores.get(node_id, 0.0)
            semantic = (
                _cosine(query_vector, self.vectors.get(node_id, []))
                if query_vector and node_id in candidate_set else 0.0
            )
            score = semantic * 0.84 + lexical * 0.16 if semantic else lexical
            scored.append({
                "id": node_id,
                "score": round(score, 6),
                "semantic": round(semantic, 6),
                "lexical": round(lexical, 6),
            })

        scored.sort(key=lambda item: item["score"], reverse=True)
        selected = [item for item in scored[:limit] if item["score"] >= settings.retrieval_threshold]
        if not selected and scored and scored[0]["score"] > 0:
            selected = scored[: min(3, limit)]
        return selected


_instance: Optional[EmbeddingStore] = None


def get_embedding_store(graph: GraphStore) -> EmbeddingStore:
    global _instance
    if _instance is None or _instance.graph is not graph:
        _instance = EmbeddingStore(graph)
    return _instance
