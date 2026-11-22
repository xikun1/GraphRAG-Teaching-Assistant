"""
多模态知识图谱 RAG 后端模块

该包将原本简陋的「前端字符串包含匹配」升级为对齐 MMGraphRAG 的服务端语义检索：

    语义向量检索 (Embedding + 余弦相似度)
        → 召回 Top-K 实体 + 相关关系 + 关联图像
        → 文本 LLM 生成初步回答
        → 多模态 LLM 逐图分析
        → 融合多模态结果
        → 生成最终回答 (流式输出)

子模块：
    config          : 配置（读取 .env 与默认参数）
    llm             : 异步 LLM/MLLM/Embedding 客户端（基于 httpx，支持流式）
    graph_store     : GraphML 加载、去引号、缓存
    embedding_store : 节点向量构建/缓存与余弦 Top-K 检索
    prompts         : 中文 RAG 提示词
    retriever       : 本地上下文构建（实体 + 关系 + 图像）
    pipeline        : 多阶段编排（异步生成器，产出 SSE 事件）
"""

from .config import settings

__all__ = ["settings"]
