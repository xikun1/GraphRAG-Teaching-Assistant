"""Validate that a GraphRAG checkout is complete and self-contained."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GRAPH = DATA / "graph" / "example_mmkg.graphml"
IMAGE_META = DATA / "metadata" / "kv_store_image_data.json"
TEXT_CHUNKS = DATA / "metadata" / "kv_store_text_chunks.json"
IMAGES = DATA / "images"
RESOURCE_ROOT = ROOT / "static" / "resources"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)
    print(f"[失败] {message}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    errors: list[str] = []
    for path in (GRAPH, IMAGE_META, TEXT_CHUNKS, IMAGES, ROOT / "static" / "index.html"):
        if not path.exists():
            fail(f"缺少必要文件或目录：{path.relative_to(ROOT)}", errors)

    if errors:
        return 1

    try:
        root = ET.parse(GRAPH).getroot()
        node_count = sum(1 for item in root.iter() if local_name(item.tag) == "node")
        edge_count = sum(1 for item in root.iter() if local_name(item.tag) == "edge")
    except (ET.ParseError, OSError) as exc:
        fail(f"GraphML 无法解析：{exc}", errors)
        node_count = edge_count = 0

    try:
        image_meta = json.loads(IMAGE_META.read_text(encoding="utf-8"))
        if not isinstance(image_meta, dict):
            raise ValueError("根对象不是 JSON object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        fail(f"图片元数据无法解析：{exc}", errors)
        image_meta = {}

    try:
        chunks = json.loads(TEXT_CHUNKS.read_text(encoding="utf-8"))
        if not isinstance(chunks, dict):
            raise ValueError("根对象不是 JSON object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        fail(f"文本块数据无法解析：{exc}", errors)
        chunks = {}

    image_files = {
        path.name.lower(): path
        for path in IMAGES.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    missing_images: list[str] = []
    nonportable_paths: list[str] = []
    for record in image_meta.values():
        if not isinstance(record, dict):
            continue
        value = str(record.get("image_path", "")).replace("\\", "/")
        if value and not value.startswith("data/images/"):
            nonportable_paths.append(value)
        filename = Path(value).name.lower()
        if filename and filename not in image_files:
            missing_images.append(filename)

    if missing_images:
        fail(f"元数据引用了 {len(set(missing_images))} 张缺失图片", errors)
    if nonportable_paths:
        fail(f"发现 {len(nonportable_paths)} 条非项目内图片路径", errors)

    runtime_files = [ROOT / "main.py", *sorted((ROOT / "rag").glob("*.py"))]
    leaked_dependencies = []
    for path in runtime_files:
        text = path.read_text(encoding="utf-8")
        if "MMGraphRAG-main" in text or "MMGRAPHRAG_ROOT" in text:
            leaked_dependencies.append(str(path.relative_to(ROOT)))
    if leaked_dependencies:
        fail("运行代码仍包含外部 MMGraphRAG 依赖：" + ", ".join(leaked_dependencies), errors)

    resource_count = sum(1 for path in RESOURCE_ROOT.rglob("*") if path.is_file()) if RESOURCE_ROOT.exists() else 0
    print(f"GraphML：{node_count} 个原始节点，{edge_count} 条原始关系")
    print(f"图片：{len(image_files)} 个文件，{len(image_meta)} 条元数据")
    print(f"文本块：{len(chunks)} 条")
    print(f"扩展资源：{resource_count} 个文件")
    print(f"项目根目录：{ROOT}")

    if errors:
        print(f"校验未通过，共 {len(errors)} 个问题。")
        return 1
    print("校验通过：项目数据完整，运行代码不依赖外部 MMGraphRAG-main。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())