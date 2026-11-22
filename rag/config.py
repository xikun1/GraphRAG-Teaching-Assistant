"""Self-contained GraphRAG runtime configuration.

All runtime data defaults to paths inside this project. Environment variables
may override individual locations for advanced deployments, but no sibling
project is discovered or required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool = False) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _path(value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


DATA_ROOT = _path(_env("DATA_DIR", "data"))
DEFAULT_GRAPHML = DATA_ROOT / "graph" / "example_mmkg.graphml"
DEFAULT_METADATA_DIR = DATA_ROOT / "metadata"
DEFAULT_IMAGE_DIR = DATA_ROOT / "images"


@dataclass
class Settings:
    # DashScope/Qwen access. The graph explorer and local lexical retrieval work
    # without a key; generated and multimodal answers require one.
    api_key: str = field(default_factory=lambda: _env("DASHSCOPE_API_KEY") or _env("QWEN_API_KEY"))
    base_url: str = field(default_factory=lambda: _env(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).rstrip("/"))
    native_base_url: str = field(default_factory=lambda: _env(
        "DASHSCOPE_NATIVE_BASE_URL", "https://dashscope.aliyuncs.com"
    ).rstrip("/"))

    text_model: str = field(default_factory=lambda: _env("TEXT_MODEL", "qwen3.7-plus"))
    mm_model: str = field(default_factory=lambda: _env("MM_MODEL", "qwen3.7-plus"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "qwen3-vl-embedding"))
    embed_dim: int = field(default_factory=lambda: _int("EMBED_DIM", 1024))
    embed_batch_size: int = field(default_factory=lambda: _int("EMBED_BATCH_SIZE", 8))
    embed_candidate_pool: int = field(default_factory=lambda: _int("EMBED_CANDIDATE_POOL", 36))

    # Portable data layout. Every default resolves below PROJECT_ROOT/data.
    data_dir: Path = field(default_factory=lambda: DATA_ROOT)
    graphml_path: Path = field(default_factory=lambda: _path(_env("GRAPHML_PATH", str(DEFAULT_GRAPHML))))
    image_data_path: Path = field(default_factory=lambda: _path(_env(
        "IMAGE_DATA_PATH", str(DEFAULT_METADATA_DIR / "kv_store_image_data.json")
    )))
    text_chunks_path: Path = field(default_factory=lambda: _path(_env(
        "TEXT_CHUNKS_PATH", str(DEFAULT_METADATA_DIR / "kv_store_text_chunks.json")
    )))
    image_dir: Path = field(default_factory=lambda: _path(_env("IMAGE_DIR", str(DEFAULT_IMAGE_DIR))))
    cache_dir: Path = field(default_factory=lambda: _path(_env("CACHE_DIR", str(DATA_ROOT / "cache"))))
    upload_dir: Path = field(default_factory=lambda: _path(_env("UPLOAD_DIR", str(DATA_ROOT / "uploads"))))
    backup_dir: Path = field(default_factory=lambda: _path(_env("BACKUP_DIR", str(DATA_ROOT / "backups"))))

    top_k: int = field(default_factory=lambda: _int("TOP_K", 8))
    retrieval_threshold: float = field(default_factory=lambda: _float("RETRIEVAL_THRESHOLD", 0.18))
    number_of_mmentities: int = field(default_factory=lambda: _int("NUMBER_OF_MMENTITIES", 4))
    image_retrieval_threshold: float = field(default_factory=lambda: _float("IMAGE_RETRIEVAL_THRESHOLD", 0.14))
    image_relative_threshold: float = field(default_factory=lambda: _float("IMAGE_RELATIVE_THRESHOLD", 0.72))
    image_min_lexical_score: float = field(default_factory=lambda: _float("IMAGE_MIN_LEXICAL_SCORE", 0.04))
    max_relations: int = field(default_factory=lambda: _int("MAX_RELATIONS", 18))
    max_chunks: int = field(default_factory=lambda: _int("MAX_CHUNKS", 3))
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 12))
    temperature: float = field(default_factory=lambda: _float("TEMPERATURE", 0.2))
    request_timeout: float = field(default_factory=lambda: _float("REQUEST_TIMEOUT", 180.0))
    max_output_tokens: int = field(default_factory=lambda: _int("MAX_OUTPUT_TOKENS", 1600))
    allow_graph_write: bool = field(default_factory=lambda: _bool("ALLOW_GRAPH_WRITE", False))
    response_type: str = field(default_factory=lambda: _env(
        "RESPONSE_TYPE", "结构清晰、专业且有依据的中文 Markdown 回答"
    ))

    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _int("PORT", 8000))

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    @property
    def emb_cache_path(self) -> Path:
        return self.cache_dir / f"{self.graphml_path.stem}_{self.embed_model}_{self.embed_dim}.json"


settings = Settings()
