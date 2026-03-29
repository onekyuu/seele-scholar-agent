# seele-scholar-agent

基于 LangGraph 的学术论文写作智能体。自动生成结构化论文大纲、撰写章节、支持人工审核与多轮修订。

## 功能特性

- **研究检索**：从 ArXiv 和 Semantic Scholar 搜索相关论文
- **大纲规划**：基于检索结果生成结构化论文大纲
- **章节撰写**：结合 RAG 上下文自动撰写论文章节
- **审核修订**：人工审核机制，支持多轮修改
- **多模型支持**：支持 OpenAI、DeepSeek、Groq 及任何 OpenAI 兼容 API

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-org/seele-scholar-agent.git
cd seele-scholar-agent

# 使用 uv 安装（推荐）
uv sync

# 或使用 pip 安装
pip install -e .
```

## 配置

复制环境变量文件并配置你的 API 密钥：

```bash
cp src/seele_scholar_agent/.env.example src/seele_scholar_agent/.env
```

### 必需配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | LLM 提供商的 API 密钥 | 必填 |
| `OPENAI_MODEL` | 模型名称 | `gpt-4o` |
| `OPENAI_BASE_URL` | API 端点 | `https://api.openai.com/v1` |

### 可选配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_TEMPERATURE` | LLM 温度参数 | `0.7` |
| `OPENAI_MAX_TOKENS` | 最大响应 token 数 | `4096` |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API 密钥（可选） | 空 |
| `QDRANT_URL` | Qdrant 向量数据库地址 | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API 密钥 | 空 |
| `QDRANT_COLLECTION` | Qdrant 集合名称 | `user_documents` |
| `EMBEDDING_MODEL` | 嵌入模型 | `text-embedding-3-small` |
| `MAX_REVISIONS` | 最大修订轮次 | `3` |
| `DEFAULT_TOP_K` | 默认搜索结果数量 | `10` |

### 模型提供商配置示例

**OpenAI（默认）：**
```env
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o"
OPENAI_BASE_URL = "https://api.openai.com/v1"
```

**DeepSeek：**
```env
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "deepseek-chat"
OPENAI_BASE_URL = "https://api.deepseek.com/v1"
```

**Groq（免费 Llama）：**
```env
OPENAI_API_KEY = "gsk_..."
OPENAI_MODEL = "llama-3.1-70b-versatile"
OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
```

## 使用方法

### 快速开始

```python
import asyncio
from datetime import datetime
from uuid import uuid4

from langchain_openai import ChatOpenAI
from seele_scholar_agent.config import settings
from seele_scholar_agent.graph import create_writing_graph
from seele_scholar_agent.state import AgentState


async def main():
    # 创建 LLM
    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY or None,
        base_url=settings.OPENAI_BASE_URL,
        temperature=settings.OPENAI_TEMPERATURE,
    )

    # 创建图
    app = create_writing_graph(model=model)

    # 准备初始状态
    initial_state: AgentState = {
        "thread_id": str(uuid4()),
        "topic": "你的研究主题",
        "created_at": datetime.now(),
        "tenant_id": None,
        "papers": [],
        "search_queries": [],
        "outline": None,
        "outline_approved": False,
        "sections": [],
        "current_section_index": 0,
        "sections_completed": [],
        "review_history": [],
        "current_review": None,
        "rag_context": [],
        "status": "idle",
        "pending_node": None,
        "error_message": None,
        "max_revisions": settings.MAX_REVISIONS,
        "revision_count": 0,
    }

    # 运行图
    result = await app.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": initial_state["thread_id"]}}
    )

    print(f"状态: {result.get('status')}")
    if result.get("outline"):
        print(f"标题: {result['outline'].title}")
        for s in result["outline"].sections:
            print(f"  {s.order}. {s.title}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 结合 Qdrant 使用（RAG）

```python
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

# 初始化 Qdrant 客户端
qdrant = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

# 初始化嵌入模型
embeddings = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL,
    api_key=settings.OPENAI_API_KEY or None,
)

# 创建带 RAG 支持的图
app = create_writing_graph(
    model=model,
    qdrant_client=qdrant,
    embedding_model=embeddings,
)
```

## 项目结构

```
seele_scholar_agent/
├── config.py          # 配置管理
├── state.py           # Pydantic 模型和 TypedDict 状态定义
├── graph.py           # LangGraph 工作流定义
├── logging.py         # 结构化日志配置
└── nodes/
    ├── planner.py     # 大纲规划节点
    ├── researcher.py  # 论文检索节点（ArXiv、Semantic Scholar）
    ├── writer.py      # 章节撰写节点
    ├── reviewer.py    # 人工审核节点
    └── prompts.py     # LLM 提示词
```

## AgentState 状态字段说明

`initial_state` 是 `AgentState` TypedDict，用于管理整个工作流的状态：

| 字段 | 类型 | 说明 |
|------|------|------|
| `thread_id` | `str` | 线程ID，用于对话持久化 |
| `topic` | `str` | 研究主题 |
| `created_at` | `datetime` | 创建时间 |
| `tenant_id` | `str \| None` | 租户ID（多租户场景使用） |
| `papers` | `list[PaperMetadata]` | 检索到的论文列表 |
| `search_queries` | `list[str]` | 搜索查询记录 |
| `outline` | `OutlineStructure \| None` | 生成的大纲结构 |
| `outline_approved` | `bool` | 大纲是否已审核通过 |
| `sections` | `list[SectionDraft]` | 拆分的章节列表 |
| `current_section_index` | `int` | 当前正在撰写的章节索引 |
| `sections_completed` | `list[str]` | 已完成的章节标题列表 |
| `review_history` | `list[dict]` | 审核历史记录 |
| `current_review` | `ReviewResult \| None` | 当前审核结果 |
| `rag_context` | `list[DocumentChunk]` | RAG 检索到的上下文 |
| `status` | `Literal[...]` | 当前状态 |
| `pending_node` | `str \| None` | 待处理的节点名称 |
| `error_message` | `str \| None` | 错误信息 |
| `max_revisions` | `int` | 最大修订次数 |
| `revision_count` | `int` | 当前修订计数 |

**status 可选值：** `idle` \| `researching` \| `planning` \| `writing` \| `reviewing` \| `waiting_human` \| `completed` \| `failed`

## 工作流程

```
START → researcher → planner → [人工确认] → writer → reviewer
                                                        ↓
                                                   [审核通过?]
                                                        ↓
                                              writer(下一节) 或 结束
```

1. **Researcher**：从 ArXiv 和 Semantic Scholar 检索相关论文
2. **Planner**：基于检索结果生成结构化论文大纲
3. **Writer**：根据大纲和 RAG 上下文撰写各个章节
4. **Reviewer**：人工审核；批准或请求修订

### 断点与恢复执行

graph 在 `planner` 节点后设置了断点（`interrupt_after=["planner"]`），暂停等待人工确认。

**人工确认模式：**

```python
# 第一次调用，运行到断点
result = await app.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})

# 状态变为 waiting_human，等待用户确认大纲
if result["status"] == "waiting_human":
    print(f"生成的大纲: {result['outline'].title}")

    # 用户确认后，更新状态并继续
    app.update_state(config, {"outline_approved": True})
    result = await app.ainvoke(None, config=config)  # 继续执行
```

**自动确认模式（测试用）：**

```python
# 初始状态设置 outline_approved = True，跳过人工确认
initial_state["outline_approved"] = True
result = await app.ainvoke(initial_state, config=config)  # 完整流程
```

## 开发

```bash
# 安装开发依赖
uv sync --extra dev

# 运行代码检查
ruff check src/

# 运行类型检查
mypy src/

# 运行测试
pytest
```

## 许可证

MIT
