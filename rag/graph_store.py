"""Normalized, reloadable access to MMGraphRAG training output."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import quote

import networkx as nx

from .config import PROJECT_ROOT, settings

logger = logging.getLogger("rag.graph_store")
GRAPH_FIELD_SEP = "<SEP>"
IMAGE_TYPES = {"ORI_IMG", "IMG", "IMAGE"}

TYPE_ALIASES = {
    "CONCEPT": "基础概念",
    "KNOWLEDGE_CONCEPT": "基础概念",
    "事件/基础概念": "基础概念",
    "基础概念/事件": "基础概念",
}

RELATION_ALIASES = {
    "检测与测距": "检测",
    "包括": "包含",
    "涵盖": "包含",
    "应用在": "应用于",
    "来自图像": "提取自",
    "源自图片": "提取自",
    "制作/发布": "发布",
    "不直接相关": "弱相关",
}
RELATION_WORDS = (
    "提取自", "应用于", "图像块", "弱相关", "属于", "包含", "组成", "展示",
    "解释", "对比", "导致", "依赖", "支撑", "输入", "输出", "生成", "前置",
    "后续", "位于", "采用", "推出", "发布", "制作", "检测", "测距", "相关",
)


def clean(value: Any) -> str:
    """Remove MMGraphRAG's optional outer quotes without changing content."""
    if value is None:
        return ""
    text = str(value).strip()
    while len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    return text


def clean_prose(value: Any) -> str:
    """Make merged descriptions readable and hide internal extraction wording."""
    text = clean(value)
    if not text:
        return ""
    parts: List[str] = []
    seen: Set[str] = set()
    for raw in text.split(GRAPH_FIELD_SEP):
        part = clean(raw).replace("chunk_text", "课件原文").replace("chunk text", "课件原文")
        key = "".join(part.split()).lower()
        if part and key not in seen:
            seen.add(key)
            parts.append(part)
    return "\n".join(parts)


def _float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_relation(value: str) -> str:
    """Return a stable short edge label (at most four characters)."""
    relation = clean(value) or "相关"
    relation = RELATION_ALIASES.get(relation, relation)
    if len(relation) <= 4:
        return relation
    for word in RELATION_WORDS:
        if word in relation:
            return word
    return "相关"


def display_entity_type(raw_type: str, node_id: str, description: str) -> str:
    """Map internal or placeholder types to stable Chinese labels."""
    normalized = clean(raw_type)
    upper = normalized.upper()
    if upper in IMAGE_TYPES:
        return "图片"
    if normalized in TYPE_ALIASES:
        return TYPE_ALIASES[normalized]
    if upper and upper != "UNKNOWN":
        return normalized

    text = f"{node_id} {description}"
    if any(word in text for word in ("公司", "组织", "机构", "品牌", "Cruise")):
        return "组织机构"
    if "操作系统" in text or node_id.endswith("系统"):
        return "基础概念"
    if any(word in text for word in ("性能", "精度", "效率", "鲁棒", "准确率")):
        return "性能指标"
    if any(word in node_id for word in ("传感器", "摄像头", "雷达", "车辆", "设备")):
        return "产品/设备"
    if any(word in node_id for word in ("算法", "模型", "学习", "检测", "规划")):
        return "核心技术"
    return "基础概念"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("无法读取 %s: %s", path, exc)
        return {}


class GraphStore:
    def __init__(self, graphml_path: Path):
        self.graphml_path = Path(graphml_path)
        self._signature: Tuple = ()
        self._graph: nx.Graph = nx.Graph()
        self._nodes: Dict[str, dict] = {}
        self._images: Dict[str, dict] = {}
        self._chunks: Dict[str, dict] = {}
        self._raw_to_clean: Dict[str, str] = {}
        self._raw_node_count = 0
        self._raw_edge_count = 0
        self._merged_duplicates = 0

    def _current_signature(self) -> tuple:
        paths = [self.graphml_path, settings.image_data_path, settings.text_chunks_path]
        return tuple((str(path), path.stat().st_mtime_ns if path.exists() else -1) for path in paths)

    def ensure_loaded(self) -> "GraphStore":
        signature = self._current_signature()
        if signature != self._signature:
            self._load()
            self._signature = signature
        return self

    def _load(self) -> None:
        if not self.graphml_path.exists():
            logger.warning("GraphML 不存在: %s", self.graphml_path)
            self._graph = nx.Graph()
            self._nodes = {}
            self._images = {}
            self._chunks = {}
            return

        graph = nx.read_graphml(self.graphml_path)
        image_data = _read_json(settings.image_data_path)
        self._chunks = _read_json(settings.text_chunks_path)
        self._graph = graph
        self._raw_node_count = graph.number_of_nodes()
        self._raw_edge_count = graph.number_of_edges()
        self._raw_to_clean = {raw: clean(raw) for raw in graph.nodes}

        images_by_name: Dict[str, dict] = {}
        images_by_path: Dict[str, dict] = {}
        for key, value in image_data.items():
            if not isinstance(value, dict):
                continue
            record = {**value, "record_id": key}
            name = clean(value.get("image_name"))
            path = clean(value.get("image_path"))
            if name:
                images_by_name[name] = record
            if path:
                images_by_path[Path(path.replace("\\", "/")).name.lower()] = record

        nodes: Dict[str, dict] = {}
        images: Dict[str, dict] = {}
        for raw, attrs in graph.nodes(data=True):
            node_id = self._raw_to_clean[raw]
            raw_entity_type = clean(attrs.get("entity_type")) or "UNKNOWN"
            description = clean_prose(attrs.get("description"))
            source_id = clean(attrs.get("source_id"))
            path_hint = self._first_image_path(source_id)
            path_meta = images_by_path.get(Path(path_hint.replace("\\", "/")).name.lower(), {})
            is_image = raw_entity_type.upper() in IMAGE_TYPES
            entity_type = display_entity_type(raw_entity_type, node_id, description)
            candidate_record = {
                "id": node_id,
                "raw_id": raw,
                "raw_ids": [raw],
                "entity_type": entity_type,
                "raw_entity_type": raw_entity_type,
                "description": description,
                "source_id": source_id,
                "chunk_id": clean(path_meta.get("chunk_id")),
                "degree": graph.degree(raw),
                "is_image": is_image,
            }
            if node_id in nodes:
                self._merge_node_records(nodes[node_id], candidate_record)
                record = nodes[node_id]
            else:
                nodes[node_id] = candidate_record
                record = candidate_record
            if is_image:
                meta = images_by_name.get(node_id) or path_meta or {}
                image_path = clean(meta.get("image_path")) or path_hint
                resolved = self._resolve_image_path(image_path)
                image_description = (
                    clean_prose(meta.get("description"))
                    or clean_prose(meta.get("caption"))
                    or description
                )
                images[node_id] = {
                    "id": node_id,
                    "name": clean(meta.get("image_name")) or node_id,
                    "description": image_description,
                    "caption": clean_prose(meta.get("caption")),
                    "footnote": clean_prose(meta.get("footnote")),
                    "image_path": image_path,
                    "resolved_path": str(resolved) if resolved else "",
                    "source_id": source_id,
                    "chunk_id": clean(meta.get("chunk_id")) or record["chunk_id"],
                    "media_url": f"/api/media/{quote(node_id, safe='')}",
                }
                record["chunk_id"] = images[node_id]["chunk_id"]

        self._merged_duplicates = self._raw_node_count - len(nodes)
        self._nodes, self._images = nodes, images
        logger.info(
            "加载图谱：%d 节点、%d 条边、%d 张命名图片、%d 个文本块",
            len(nodes), graph.number_of_edges(), len(images), len(self._chunks),
        )

    @staticmethod
    def _record_priority(record: dict) -> int:
        source = record.get("source_id", "").lower()
        priority = 100 if "chunk" in source else 10 if source.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")) else 0
        type_priority = {
            "\u56fe\u7247": 100,
            "\u7ec4\u7ec7\u673a\u6784": 60,
            "\u4eba\u7269": 60,
            "\u5730\u7406\u4f4d\u7f6e": 50,
            "\u4e8b\u4ef6": 50,
            "\u6838\u5fc3\u6280\u672f": 45,
            "\u6027\u80fd\u6307\u6807": 40,
            "\u4ea7\u54c1/\u8bbe\u5907": 30,
            "\u57fa\u7840\u6982\u5ff5": 20,
        }
        return priority + type_priority.get(record.get("entity_type", ""), 0)

    @classmethod
    def _merge_node_records(cls, existing: dict, candidate: dict) -> None:
        """Merge quoted/unquoted aliases without losing either evidence path."""
        candidate_first = cls._record_priority(candidate) > cls._record_priority(existing)
        first, second = (candidate, existing) if candidate_first else (existing, candidate)
        descriptions = []
        for value in (first.get("description", ""), second.get("description", "")):
            if value and value not in descriptions:
                descriptions.append(value)
        sources = []
        for value in (first.get("source_id", ""), second.get("source_id", "")):
            for part in value.split(GRAPH_FIELD_SEP):
                if part and part not in sources:
                    sources.append(part)
        if candidate_first:
            for key in ("entity_type", "raw_entity_type", "chunk_id"):
                if candidate.get(key):
                    existing[key] = candidate[key]
        elif not existing.get("chunk_id") and candidate.get("chunk_id"):
            existing["chunk_id"] = candidate["chunk_id"]
        existing["description"] = "\n".join(descriptions)
        existing["source_id"] = GRAPH_FIELD_SEP.join(sources)
        existing["degree"] = existing.get("degree", 0) + candidate.get("degree", 0)
        existing["is_image"] = bool(existing.get("is_image") or candidate.get("is_image"))
        existing.setdefault("raw_ids", [existing["raw_id"]]).extend(candidate.get("raw_ids", [candidate["raw_id"]]))

    @staticmethod
    def _first_image_path(source_id: str) -> str:
        for part in source_id.split(GRAPH_FIELD_SEP):
            candidate = part.strip()
            if candidate.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                return candidate
        return source_id.strip()

    def _resolve_image_path(self, value: str) -> Optional[Path]:
        if not value:
            return None
        normalized = value.replace("\\", "/")
        raw = Path(normalized)
        # Bundled images are resolved by basename first. This deliberately
        # prevents stale absolute paths in imported metadata from becoming a
        # runtime dependency on the training workspace.
        candidates = [settings.image_dir / raw.name]
        if not raw.is_absolute():
            candidates.append(PROJECT_ROOT / raw)
        candidates.append(settings.upload_dir / raw.name)
        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
                if resolved.is_file():
                    return resolved
            except OSError:
                continue
        return None

    @property
    def nodes(self) -> Dict[str, dict]:
        return self._nodes

    @property
    def images(self) -> Dict[str, dict]:
        return self._images

    @property
    def chunks(self) -> Dict[str, dict]:
        return self._chunks

    def node(self, node_id: str) -> Optional[dict]:
        return self._nodes.get(node_id)

    def embedding_text(self, node_id: str) -> str:
        record = self._nodes.get(node_id, {})
        return (
            f"名称：{node_id}\n"
            f"类型：{record.get('entity_type', '')}\n"
            f"说明：{record.get('description', '')}"
        ).strip()

    def neighbors(self, node_id: str) -> List[str]:
        record = self._nodes.get(node_id)
        if not record:
            return []
        result: Set[str] = set()
        for raw_id in record.get("raw_ids", [record["raw_id"]]):
            result.update(self._raw_to_clean.get(raw, clean(raw)) for raw in self._graph.neighbors(raw_id))
        result.discard(node_id)
        return sorted(result, key=lambda value: self._nodes.get(value, {}).get("degree", 0), reverse=True)

    def _iter_edge_data(self, raw_id: str) -> Iterable[Tuple[str, str, dict]]:
        if self._graph.is_multigraph():
            for source, target, _, attrs in self._graph.edges(raw_id, data=True, keys=True):
                yield source, target, attrs
        else:
            for source, target, attrs in self._graph.edges(raw_id, data=True):
                yield source, target, attrs

    def _iter_all_edge_data(self) -> Iterable[Tuple[str, str, dict]]:
        if self._graph.is_multigraph():
            for source, target, _, attrs in self._graph.edges(data=True, keys=True):
                yield source, target, attrs
        else:
            for source, target, attrs in self._graph.edges(data=True):
                yield source, target, attrs

    def _edge_record(self, source: str, target: str, attrs: dict) -> dict:
        raw_relation = clean(attrs.get("relation")) or clean(attrs.get("description")) or "相关"
        relation = normalize_relation(raw_relation)
        return {
            "source": self._raw_to_clean.get(source, clean(source)),
            "target": self._raw_to_clean.get(target, clean(target)),
            "relation": relation,
            "raw_relation": raw_relation,
            "description": clean_prose(attrs.get("description")) or relation,
            "evidence": clean_prose(attrs.get("evidence")),
            "source_id": clean(attrs.get("source_id")),
            "weight": _float(attrs.get("weight")),
            "order": _int(attrs.get("order")),
            "rank": self._graph.degree(source) + self._graph.degree(target),
        }

    def edges_of(self, node_id: str) -> List[dict]:
        record = self._nodes.get(node_id)
        if not record:
            return []
        result: List[dict] = []
        seen: Set[Tuple] = set()
        for raw_id in record.get("raw_ids", [record["raw_id"]]):
            for source, target, attrs in self._iter_edge_data(raw_id):
                edge = self._edge_record(source, target, attrs)
                if edge["source"] == edge["target"]:
                    continue
                endpoints = (edge["source"], edge["target"])
                if not self._graph.is_directed():
                    endpoints = tuple(sorted(endpoints))
                key = endpoints + (edge["relation"], edge["evidence"])
                if key not in seen:
                    seen.add(key)
                    result.append(edge)
        return result

    def all_edges(self) -> List[dict]:
        """Serialize every unique edge in one pass (important for large graphs)."""
        result: List[dict] = []
        seen: Set[Tuple] = set()
        for source, target, attrs in self._iter_all_edge_data():
            edge = self._edge_record(source, target, attrs)
            if edge["source"] == edge["target"]:
                continue
            endpoints = (edge["source"], edge["target"])
            if not self._graph.is_directed():
                endpoints = tuple(sorted(endpoints))
            key = endpoints + (edge["relation"], edge["evidence"])
            if key not in seen:
                seen.add(key)
                result.append(edge)
        return result

    def associated_images(self, node_ids: List[str], limit: Optional[int] = None) -> List[dict]:
        seen: Set[str] = set()
        result: List[dict] = []
        for node_id in node_ids:
            for candidate in [node_id, *self.neighbors(node_id)]:
                if candidate in self._images and candidate not in seen:
                    seen.add(candidate)
                    result.append(self._images[candidate])
                    if limit and len(result) >= limit:
                        return result
        return result

    def source_chunks(self, node_ids: List[str], limit: int = 4) -> List[dict]:
        wanted: List[str] = []
        for node_id in node_ids:
            record = self._nodes.get(node_id, {})
            if record.get("chunk_id"):
                wanted.append(record["chunk_id"])
            source_id = record.get("source_id", "")
            wanted.extend(part.strip() for part in source_id.split(GRAPH_FIELD_SEP) if "chunk" in part.lower())
            for edge in self.edges_of(node_id):
                wanted.extend(part.strip() for part in edge["source_id"].split(GRAPH_FIELD_SEP) if "chunk" in part.lower())

        result: List[dict] = []
        seen: Set[str] = set()
        for chunk_id in wanted:
            if chunk_id in seen or chunk_id not in self._chunks:
                continue
            seen.add(chunk_id)
            value = self._chunks[chunk_id]
            content = value.get("content", "") if isinstance(value, dict) else str(value)
            result.append({"id": chunk_id, "content": clean_prose(content)})
            if len(result) >= limit:
                break
        return result

    def media_path(self, image_id: str) -> Optional[Path]:
        record = self._images.get(image_id)
        if not record:
            return None
        value = record.get("resolved_path")
        path = Path(value) if value else None
        return path if path and path.is_file() else None

    def graph_payload(self) -> dict:
        nodes = []
        for node_id, record in self._nodes.items():
            item = {key: value for key, value in record.items() if key not in {"raw_id", "raw_ids"}}
            if node_id in self._images:
                image = self._images[node_id]
                item.update({
                    "name": image["name"],
                    "media_url": image["media_url"],
                    "caption": image.get("caption", ""),
                    "footnote": image.get("footnote", ""),
                })
            nodes.append(item)
        edges = self.all_edges()
        type_counts = Counter(item["entity_type"] for item in nodes)
        relation_counts = Counter(item["relation"] for item in edges)
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "nodes": len(nodes),
                "raw_nodes": self._raw_node_count,
                "merged_duplicates": self._merged_duplicates,
                "edges": len(edges),
                "raw_edges": self._raw_edge_count,
                "images": len(self._images),
                "chunks": len(self._chunks),
                "types": dict(sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))),
                "relations": len(relation_counts),
            },
            "source": str(self.graphml_path),
        }


_store: Optional[GraphStore] = None


def get_store() -> GraphStore:
    global _store
    if _store is None or _store.graphml_path != settings.graphml_path:
        _store = GraphStore(settings.graphml_path)
    return _store.ensure_loaded()
