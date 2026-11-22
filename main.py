"""FastAPI entry point for the multimodal GraphRAG explorer."""

from __future__ import annotations

import json
import logging
import re
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.gzip import GZipMiddleware

from rag.config import PROJECT_ROOT, settings
from rag.graph_store import clean, get_store, normalize_relation
from rag.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("graphrag")

app = FastAPI(title="Multimodal GraphRAG", version="4.0.0")
app.add_middleware(GZipMiddleware, minimum_size=1000)
STATIC_DIR = PROJECT_ROOT / "static"
RESOURCE_DIR = STATIC_DIR / "resources"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if RESOURCE_DIR.is_dir():
    app.mount("/resources", StaticFiles(directory=RESOURCE_DIR), name="resources")

GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
MAX_GRAPHML_BYTES = 64 * 1024 * 1024
MAX_BACKUPS = 12


class ChatRequest(BaseModel):
    query: str = Field(default="", max_length=12000)
    image_base64_list: List[str] = Field(default_factory=list)
    image_base64: Optional[str] = None
    attachment_texts: List[str] = Field(default_factory=list)

    def images(self) -> List[str]:
        values = list(self.image_base64_list)
        if self.image_base64:
            values.insert(0, self.image_base64)
        return [item for item in values if item][:4]


class GraphSaveRequest(BaseModel):
    graphml: str = Field(min_length=50)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _normalize_saved_graph(root: ET.Element) -> Tuple[int, int, int]:
    """Validate GraphML structure and normalize every relation to a short word."""
    if _local_name(root.tag) != "graphml":
        raise HTTPException(400, "提交内容不是 GraphML 文档")

    key_names = {}
    for element in root.iter():
        if _local_name(element.tag) == "key":
            key_names[element.attrib.get("id", "")] = element.attrib.get("attr.name", "")

    node_count = 0
    edge_count = 0
    normalized_count = 0
    for element in root.iter():
        name = _local_name(element.tag)
        if name == "node":
            node_count += 1
        elif name == "edge":
            edge_count += 1
        elif name == "data" and key_names.get(element.attrib.get("key", "")) == "relation":
            original = element.text or ""
            stripped = original.strip()
            was_quoted = len(stripped) >= 2 and stripped[0] == stripped[-1] == '"'
            relation = normalize_relation(clean(original))
            replacement = f'"{relation}"' if was_quoted else relation
            if replacement != original:
                element.text = replacement
                normalized_count += 1

    if node_count == 0:
        raise HTTPException(400, "GraphML 中没有节点，已拒绝覆盖现有图谱")
    return node_count, edge_count, normalized_count


def _backup_graph(target: Path) -> None:
    if not target.exists():
        return
    backup_dir = settings.backup_dir
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = backup_dir / f"{target.stem}-{stamp}.graphml"
    backup.write_bytes(target.read_bytes())
    backups = sorted(backup_dir.glob(f"{target.stem}-*.graphml"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for old in backups[MAX_BACKUPS:]:
        try:
            old.unlink()
        except OSError:
            logger.warning("无法清理旧备份: %s", old)


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/graph")
async def graph_json():
    return get_store().graph_payload()


@app.get("/api/resources")
async def resources():
    categories = {"论文": "papers", "书籍": "books", "专利": "patents"}
    result = {}
    for label, folder in categories.items():
        directory = RESOURCE_DIR / folder
        if not directory.is_dir() or directory.is_symlink():
            continue
        result[label] = [
            {"title": path.stem, "file": f"/static/resources/{folder}/{quote(path.name, safe='')}"}
            for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold())
            if path.suffix.lower() == ".pdf" and path.is_file() and not path.is_symlink()
        ]
    return result


@app.get("/api/graphml")
async def graphml():
    if not settings.graphml_path.is_file():
        raise HTTPException(404, f"GraphML 不存在：{settings.graphml_path}")
    return FileResponse(settings.graphml_path, media_type="application/xml", filename=settings.graphml_path.name)


@app.get("/api/media/{image_id:path}")
async def media(image_id: str):
    path = get_store().media_path(image_id)
    if not path:
        raise HTTPException(404, "图片不存在或未被图谱引用")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=3600"})


@app.post("/api/chat")
async def chat(request: ChatRequest):
    images = request.images()
    if not request.query.strip() and not images:
        raise HTTPException(400, "请输入问题或上传图片")
    encoded_size = sum(len(item) for item in images)
    if encoded_size > settings.max_upload_mb * 1024 * 1024 * 4 * 4 // 3:
        raise HTTPException(413, "问答图片总体积过大，请压缩后重试")

    async def events():
        async for event in run_pipeline(request.query, images, request.attachment_texts):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/chat_simple")
async def chat_simple(request: ChatRequest):
    answer, meta = "", {}
    async for event in run_pipeline(request.query, request.images(), request.attachment_texts):
        if event.get("event") == "token":
            answer += event.get("text", "")
        elif event.get("event") in {"retrieval", "done"}:
            meta[event["event"]] = event
        elif event.get("event") == "error":
            raise HTTPException(500, event.get("message", "未知错误"))
    return {"answer": answer, "meta": meta}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    mime = (file.content_type or "").lower()
    if not mime.startswith("image/"):
        raise HTTPException(415, "仅支持图片上传")
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"图片不能超过 {settings.max_upload_mb} MB")
    suffix = Path(file.filename or "image.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        suffix = ".jpg"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    target = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    target.write_bytes(content)
    return {"filename": target.name, "url": f"/api/uploads/{target.name}"}


@app.get("/api/uploads/{filename}")
async def uploaded_file(filename: str):
    if not re.fullmatch(r"[0-9a-f]{32}\.(jpg|jpeg|png|webp|gif)", filename, re.IGNORECASE):
        raise HTTPException(400, "非法文件名")
    path = (settings.upload_dir / filename).resolve()
    if path.parent != settings.upload_dir.resolve() or not path.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=3600"})


@app.delete("/api/uploads/{filename}")
async def delete_uploaded_file(filename: str):
    path = (settings.upload_dir / Path(filename).name).resolve()
    if path.parent != settings.upload_dir.resolve():
        raise HTTPException(400, "非法文件名")
    if path.exists():
        path.unlink()
    return {"ok": True}


@app.post("/api/save_graphml")
async def save_graphml(request: GraphSaveRequest):
    if not settings.allow_graph_write:
        raise HTTPException(403, "图谱写回已关闭；如需启用，请设置 ALLOW_GRAPH_WRITE=true")
    encoded = request.graphml.encode("utf-8")
    if len(encoded) > MAX_GRAPHML_BYTES:
        raise HTTPException(413, "GraphML 超过 64 MB，已拒绝写入")
    try:
        root = ET.fromstring(encoded)
    except ET.ParseError as exc:
        raise HTTPException(400, f"GraphML XML 无效：{exc}") from exc
    nodes, edges, normalized = _normalize_saved_graph(root)

    ET.register_namespace("", GRAPHML_NS)
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    target = settings.graphml_path
    target.parent.mkdir(parents=True, exist_ok=True)
    _backup_graph(target)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(target)
    logger.info("图谱写回完成：%d 节点、%d 条边、%d 条关系已归一化", nodes, edges, normalized)
    return {
        "ok": True,
        "path": str(target),
        "nodes": nodes,
        "edges": edges,
        "normalized_relations": normalized,
    }


def _inside_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
        return True
    except (OSError, ValueError):
        return False


@app.get("/api/health")
async def health():
    store = get_store()
    stats = store.graph_payload()["stats"]
    runtime_paths = (
        settings.graphml_path,
        settings.image_data_path,
        settings.text_chunks_path,
        settings.image_dir,
    )
    return {
        "status": "ok" if store.nodes else "degraded",
        "runtime_mode": "self-contained",
        "portable_paths": all(_inside_project(path) for path in runtime_paths),
        "api_key_configured": settings.has_key,
        "models": {"chat": settings.text_model, "vision": settings.mm_model, "embedding": settings.embed_model},
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "data_root": str(settings.data_dir),
            "graphml": str(settings.graphml_path),
            "graphml_exists": settings.graphml_path.exists(),
            "image_data": str(settings.image_data_path),
            "text_chunks": str(settings.text_chunks_path),
            "images": str(settings.image_dir),
            "cache": str(settings.cache_dir),
            "uploads": str(settings.upload_dir),
            "backups": str(settings.backup_dir),
        },
        "graph": stats,
        "graph_write_enabled": settings.allow_graph_write,
        "embedding_strategy": {"mode": "lazy-candidate", "candidate_pool": settings.embed_candidate_pool},
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
