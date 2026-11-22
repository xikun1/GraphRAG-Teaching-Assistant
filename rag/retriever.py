"""Graph-aware retrieval and source-grounded context assembly."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from .config import settings
from .embedding_store import EmbeddingStore, _lexical_score
from .graph_store import GraphStore


def _rank_associated_images(
    query: str,
    graph: GraphStore,
    ordered_ids: List[str],
    scores: Dict[str, float],
    hit_ids: Set[str],
    has_query_image: bool,
) -> List[dict]:
    """Rank graph images by retrieval evidence instead of adjacency order.

    Text questions require a small amount of image-text overlap so that a very
    broad high-scoring entity cannot pull in unrelated neighboring images. For
    uploaded-image questions the multimodal retrieval score is trusted more.
    """
    max_images = max(0, settings.number_of_mmentities)
    if not max_images:
        return []

    candidates: Dict[str, dict] = {}
    lexical_cache: Dict[str, float] = {}
    for node_id in ordered_ids:
        parent_score = max(0.0, float(scores.get(node_id, 0.0)))
        for candidate_id in [node_id, *graph.neighbors(node_id)]:
            image = graph.images.get(candidate_id)
            if not image:
                continue

            lexical = lexical_cache.setdefault(
                candidate_id,
                _lexical_score(query, graph.embedding_text(candidate_id)),
            )
            direct_hit = candidate_id in hit_ids
            if (
                not has_query_image
                and not direct_hit
                and lexical < settings.image_min_lexical_score
            ):
                continue

            candidate_score = max(
                float(scores.get(candidate_id, 0.0)),
                parent_score if candidate_id == node_id else parent_score * 0.90,
            )
            state = candidates.setdefault(candidate_id, {
                "record": image,
                "association_score": 0.0,
                "lexical_score": lexical,
                "direct_hit": False,
                "matched_via": node_id,
                "support": set(),
            })
            state["support"].add(node_id)
            state["direct_hit"] = state["direct_hit"] or direct_hit
            if candidate_score > state["association_score"]:
                state["association_score"] = candidate_score
                state["matched_via"] = candidate_id if direct_hit else node_id

    ranked: List[dict] = []
    for state in candidates.values():
        association = state["association_score"]
        lexical = state["lexical_score"]
        if state["direct_hit"]:
            score = association * 0.55 + lexical * 0.45 + 0.04
        elif has_query_image:
            score = association * 0.62 + lexical * 0.38
        else:
            score = association * 0.45 + lexical * 0.55
        score += min(0.04, max(0, len(state["support"]) - 1) * 0.01)

        item = dict(state["record"])
        item.update({
            "score": round(min(1.0, max(0.0, score)), 6),
            "association_score": round(association, 6),
            "lexical_score": round(lexical, 6),
            "matched_via": state["matched_via"],
            "direct_hit": state["direct_hit"],
            "support_count": len(state["support"]),
            "available": bool(state["record"].get("resolved_path")),
        })
        ranked.append(item)

    ranked.sort(
        key=lambda item: (item["score"], item["available"], item["support_count"]),
        reverse=True,
    )
    if not ranked:
        return []

    best_score = ranked[0]["score"]
    absolute_threshold = max(0.0, settings.image_retrieval_threshold)
    relative_threshold = min(1.0, max(0.0, settings.image_relative_threshold))
    cutoff = max(absolute_threshold, best_score * relative_threshold)
    selected = [item for item in ranked if item["score"] >= cutoff]
    if not selected and best_score >= absolute_threshold * 0.85:
        selected = ranked[:1]
    return selected[:max_images]


async def build_context(
    query: str,
    graph: GraphStore,
    embeddings: EmbeddingStore,
    images: Optional[List[str]] = None,
) -> dict:
    images = images or []
    hits = await embeddings.search(query, images=images)
    scores = {hit["id"]: hit["score"] for hit in hits}
    hit_ids = set(scores)

    # Controlled one-hop expansion retains graph structure without letting a
    # single high-degree node take over the answer context.
    for hit in hits[:5]:
        for neighbor in graph.neighbors(hit["id"])[:6]:
            scores.setdefault(neighbor, round(hit["score"] * 0.72, 6))

    ordered_ids = sorted(scores, key=scores.get, reverse=True)[: max(settings.top_k, 10)]
    entities = []
    for node_id in ordered_ids:
        record = graph.node(node_id)
        if not record:
            continue
        entities.append({
            "name": node_id,
            "type": record["entity_type"],
            "description": record["description"],
            "score": scores[node_id],
            "is_image": record["is_image"],
        })

    edge_map = {}
    for node_id in ordered_ids:
        for edge in graph.edges_of(node_id):
            key = tuple(sorted((edge["source"], edge["target"]))) + (edge["relation"], edge["evidence"])
            edge["retrieval_score"] = max(scores.get(edge["source"], 0), scores.get(edge["target"], 0))
            edge_map[key] = edge
    edges = sorted(
        edge_map.values(),
        key=lambda item: (item["retrieval_score"], item["weight"], item["rank"]),
        reverse=True,
    )[: settings.max_relations]
    graph_images = _rank_associated_images(
        query,
        graph,
        ordered_ids,
        scores,
        hit_ids,
        bool(images),
    )
    chunks = graph.source_chunks(ordered_ids, settings.max_chunks)

    lines = ["# 检索到的图谱实体"]
    for entity in entities:
        description = (entity["description"] or "无")[:650]
        lines.append(
            f"- [实体：{entity['name']}] 类型={entity['type']}；说明={description}；相关度={entity['score']:.3f}"
        )
    lines.append("\n# 关系（短关系词与证据分离）")
    for edge in edges:
        evidence = (edge["evidence"] or edge["description"] or "无")[:600]
        lines.append(
            f"- [实体：{edge['source']}] --{edge['relation']}--> [实体：{edge['target']}]；证据：{evidence}"
        )
    lines.append("\n# 候选关联图像（已按相关度动态筛选）")
    for image in graph_images:
        lines.append(
            f"- [图像：{image['name']}] 相关度={image['score']:.3f}；"
            f"关联实体={image['matched_via']}；{(image['description'] or '无描述')[:650]}"
        )
    lines.append("\n# 课件原文证据")
    for chunk in chunks:
        lines.append(f"- [课件原文] {chunk['content'][:1400]}")

    return {
        "found": bool(entities),
        "entities": entities,
        "edges": edges,
        "images": graph_images,
        "chunks": chunks,
        "context": "\n".join(lines),
        "retrieval_mode": embeddings.mode,
        "retrieval_error": embeddings.last_error,
    }
