"""Async DashScope clients for chat and Qwen3-VL multimodal embeddings."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, Dict, List, Optional

import httpx

from .config import settings


class LLMError(RuntimeError):
    pass


def _headers() -> Dict[str, str]:
    if not settings.api_key:
        raise LLMError("未配置 DASHSCOPE_API_KEY，已切换到本地检索模式")
    return {"Authorization": f"Bearer {settings.api_key}", "Content-Type": "application/json"}


async def _post_json(url: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        try:
            response = await client.post(url, headers=_headers(), json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            message = exc.response.text[:500]
            raise LLMError(f"模型接口返回 {exc.response.status_code}: {message}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"模型接口请求失败: {exc}") from exc


def _embedding_from_response(data: dict) -> List[float]:
    output = data.get("output", {})
    values = output.get("embeddings") or data.get("data") or []
    if values and isinstance(values[0], dict):
        vector = values[0].get("embedding")
        if isinstance(vector, list):
            return vector
    raise LLMError(f"无法解析多模态向量响应: {str(data)[:300]}")


async def embed_multimodal(text: str = "", images: Optional[List[str]] = None) -> List[float]:
    """Create one fused vector from text and zero or more image data URIs/URLs."""
    contents: List[dict] = []
    if text.strip():
        contents.append({"text": text.strip()})
    for image in images or []:
        if image:
            contents.append({"image": image})
    if not contents:
        return []
    payload = {
        "model": settings.embed_model,
        "input": {"contents": contents},
        "parameters": {"dimension": settings.embed_dim, "enable_fusion": True},
    }
    url = f"{settings.native_base_url}/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
    return _embedding_from_response(await _post_json(url, payload))


async def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    semaphore = asyncio.Semaphore(max(1, min(settings.embed_batch_size, 8)))

    async def one(text: str) -> List[float]:
        async with semaphore:
            return await embed_multimodal(text=text)

    return list(await asyncio.gather(*(one(text) for text in texts)))


def _messages(user_prompt: str, system_prompt: Optional[str] = None) -> list:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _token_options(model: str) -> Dict[str, int]:
    if model == "qwen4-max":
        # Qwen4-Max counts reasoning and the visible answer together.
        thinking_budget = 1024
        return {
            "thinking_budget": thinking_budget,
            "max_completion_tokens": thinking_budget + settings.max_output_tokens,
        }
    return {"max_tokens": settings.max_output_tokens}


def _completion_text(data: dict) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("无法解析模型回答") from exc
    if not isinstance(content, str) or not content.strip():
        raise LLMError("模型未返回可显示的答案")
    return content


async def chat(user_prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> str:
    payload = {
        "model": settings.text_model,
        "messages": _messages(user_prompt, system_prompt),
        "temperature": settings.temperature if temperature is None else temperature,
        **_token_options(settings.text_model),
    }
    data = await _post_json(f"{settings.base_url}/chat/completions", payload)
    return _completion_text(data)


async def chat_stream(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
) -> AsyncGenerator[str, None]:
    payload = {
        "model": settings.text_model,
        "messages": _messages(user_prompt, system_prompt),
        "temperature": settings.temperature if temperature is None else temperature,
        **_token_options(settings.text_model),
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        async with client.stream("POST", f"{settings.base_url}/chat/completions", headers=_headers(), json=payload) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise LLMError(f"模型流式接口返回 {response.status_code}: {body[:500]}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if value == "[DONE]":
                    break
                try:
                    delta = json.loads(value)["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue


async def chat_multimodal(
    user_prompt: str,
    images: List[str],
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    content: List[dict] = [{"type": "image_url", "image_url": {"url": image}} for image in images if image]
    content.append({"type": "text", "text": user_prompt})
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    payload = {
        "model": settings.mm_model,
        "messages": messages,
        "temperature": settings.temperature if temperature is None else temperature,
        **_token_options(settings.mm_model),
    }
    data = await _post_json(f"{settings.base_url}/chat/completions", payload)
    return _completion_text(data)
