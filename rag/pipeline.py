"""End-to-end multimodal graph RAG orchestration with local fallback."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

from . import llm
from .config import settings
from .embedding_store import get_embedding_store
from .graph_store import get_store
from .prompts import ANSWER_SYSTEM_PROMPT, OFFLINE_NOTICE, VISUAL_GROUNDING_PROMPT
from .retriever import build_context

logger = logging.getLogger("rag.pipeline")


def _file_to_data_uri(path_value: str) -> Optional[str]:
    path = Path(path_value)
    if not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except OSError:
        return None


def _parse_json_object(text: str) -> dict:
    value = text.strip()
    if value.startswith("```json"):
        value = value[7:]
    elif value.startswith("```"):
        value = value[3:]
    if value.endswith("```"):
        value = value[:-3]
    value = value.strip()
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(value[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {"visual_summary": text[:800], "search_query": ""}


def _local_answer(query: str, context: dict) -> str:
    lines = [f"> {OFFLINE_NOTICE}", "", "## 本地图谱检索结果"]
    if not context["found"]:
        lines.append("没有找到与当前问题直接匹配的本地图谱节点。")
        return "\n".join(lines)
    lines.append(f"问题：{query or '图片提问'}")
    lines.append("\n### 相关实体")
    for entity in context["entities"][:6]:
        description = (entity["description"] or "\u65e0\u8865\u5145\u63cf\u8ff0")[:260]
        lines.append(f"- **{entity['name']}**\uff08{entity['type']}\uff09\uff1a{description}")
    if context["edges"]:
        lines.append("\n### 关系与证据")
        for edge in context["edges"][:8]:
            evidence = (edge["evidence"] or edge["description"] or "\u65e0\u8865\u5145\u8bc1\u636e")[:300]
            lines.append(f"- {edge['source']} \u2014 **{edge['relation']}** \u2192 {edge['target']}\uff1a{evidence}")
    if context["images"]:
        lines.append("\n### 关联图片")
        for image in context["images"]:
            description = (image["description"] or "\u65e0\u8865\u5145\u63cf\u8ff0")[:220]
            lines.append(f"- {image['name']}\uff1a{description}")
    return "\n".join(lines)


def _chunks(text: str, size: int = 40):
    for index in range(0, len(text), size):
        yield text[index:index + size]


async def run_pipeline(
    query: str,
    user_images: Optional[List[str]] = None,
    attachment_texts: Optional[List[str]] = None,
) -> AsyncGenerator[Dict, None]:
    user_images = [value for value in (user_images or []) if value][:4]
    attachment_texts = attachment_texts or []
    original_query = query.strip()
    retrieval_query = original_query
    visual_summary = ""

    try:
        if user_images and settings.has_key:
            yield {"event": "status", "stage": "vision", "label": "正在理解上传图片并生成跨模态检索词…"}
            try:
                grounded = await llm.chat_multimodal(
                    original_query or "请识别图片的主题、关键对象和可见文字，用于知识图谱检索。",
                    user_images,
                    system_prompt=VISUAL_GROUNDING_PROMPT,
                    temperature=0.0,
                )
                grounding = _parse_json_object(grounded)
                visual_summary = str(grounding.get("visual_summary", ""))
                suggested = str(grounding.get("search_query", "")).strip()
                retrieval_query = "；".join(value for value in (original_query, suggested, visual_summary) if value)
                yield {"event": "visual_grounding", "summary": visual_summary, "search_query": suggested}
            except llm.LLMError as exc:
                logger.warning("图片检索规划失败，继续用融合向量检索: %s", exc)

        if not retrieval_query:
            retrieval_query = "识别上传图片并匹配相关知识图谱内容"

        yield {"event": "status", "stage": "retrieval", "label": "正在执行候选召回、多模态检索与图结构扩展…"}
        store = get_store()
        embeddings = get_embedding_store(store)
        context = await build_context(retrieval_query, store, embeddings, user_images)
        visual_terms = ("图中", "图片", "图像", "截图", "照片", "画面", "标志", "界面")
        visual_requested = bool(user_images) or any(term in original_query for term in visual_terms)

        # Resolve only images that can actually be opened by the multimodal model.
        # The total six-image budget includes user uploads, so labels and image
        # payloads always stay aligned.
        retrieved_images: List[str] = []
        retrieved_names: List[str] = []
        if visual_requested and settings.has_key:
            graph_image_slots = max(0, 6 - len(user_images))
            for image in context["images"][: min(settings.number_of_mmentities, graph_image_slots)]:
                uri = _file_to_data_uri(image.get("resolved_path", ""))
                if uri:
                    retrieved_images.append(uri)
                    retrieved_names.append(image["name"])
        all_images = user_images + retrieved_images if visual_requested and settings.has_key else []

        yield {
            "event": "retrieval",
            "found": context["found"],
            "mode": context["retrieval_mode"],
            "warning": context["retrieval_error"],
            "entities": [
                {"name": item["name"], "type": item["type"], "score": item["score"]}
                for item in context["entities"]
                if not item["is_image"]
            ],
            "images": [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "media_url": item["media_url"],
                    "score": item["score"],
                    "matched_via": item["matched_via"],
                    "available": item["available"],
                }
                for item in context["images"]
            ],
            "chunk_count": len(context["chunks"]),
            "visual_requested": visual_requested,
            "visual_available": settings.has_key,
            "user_image_count": len(user_images),
            "planned_graph_image_count": len(retrieved_images),
        }

        if not settings.has_key:
            answer = _local_answer(original_query, context)
            for token in _chunks(answer):
                yield {"event": "token", "text": token}
            yield {"event": "done", "answer": answer, "mode": "local"}
            return

        visual_context = visual_summary
        if all_images:
            yield {"event": "status", "stage": "vision", "label": f"正在联合分析 {len(all_images)} 张问题图片与图谱图片…"}
            image_labels = (
                [f"用户图片{i + 1}" for i in range(len(user_images))]
                + [f"图谱图片：{name}" for name in retrieved_names]
            )
            try:
                visual_context = await llm.chat_multimodal(
                    f"用户问题：{original_query or '请解释图片内容'}\n图片顺序：{image_labels}\n"
                    "请只陈述与问题有关且在图片中可确认的事实，并指出对应图片名称。",
                    all_images,
                    system_prompt="你是严谨的多图联合分析器。比较用户图片与知识图谱关联图片，不确定之处明确说明。",
                )
                yield {
                    "event": "visual_analysis",
                    "success": True,
                    "analyzed_image_count": len(all_images),
                    "user_image_count": len(user_images),
                    "graph_image_count": len(retrieved_images),
                }
            except llm.LLMError as exc:
                logger.warning("联合图片分析失败: %s", exc)
                yield {
                    "event": "visual_analysis",
                    "success": False,
                    "analyzed_image_count": 0,
                    "user_image_count": len(user_images),
                    "graph_image_count": 0,
                    "message": "视觉分析失败，已回退为图谱文本证据",
                }
        elif visual_requested:
            yield {
                "event": "visual_analysis",
                "success": False,
                "analyzed_image_count": 0,
                "user_image_count": len(user_images),
                "graph_image_count": 0,
                "message": "没有可读取的候选图片，已使用图谱文本证据",
            }
        context_text = context["context"]
        if attachment_texts:
            context_text += "\n\n# 用户附件文本\n" + "\n\n".join(text[:6000] for text in attachment_texts)
        system_prompt = ANSWER_SYSTEM_PROMPT.format(context=context_text, visual_context=visual_context or "无")
        yield {"event": "status", "stage": "answer", "label": "正在基于图谱证据生成答案…"}
        answer_parts: List[str] = []
        try:
            async for token in llm.chat_stream(
                original_query or "请结合上传图片和知识图谱作答。",
                system_prompt=system_prompt,
            ):
                answer_parts.append(token)
                yield {"event": "token", "text": token}
            yield {"event": "done", "answer": "".join(answer_parts), "mode": "multimodal-rag"}
        except llm.LLMError as exc:
            logger.warning("生成失败，返回本地证据摘要: %s", exc)
            answer = _local_answer(original_query, context)
            for token in _chunks(answer):
                yield {"event": "token", "text": token}
            yield {"event": "done", "answer": answer, "mode": "local-fallback", "warning": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("多模态 RAG 管线异常")
        yield {"event": "error", "message": f"服务内部错误：{exc}"}
