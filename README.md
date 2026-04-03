# seele-scholar-agent

基于 LangGraph 的学术论文写作智能体。自动生成结构化论文大纲、撰写章节、支持人工审核与多轮修订。

## 功能特性

- **研究检索**：从 ArXiv、Semantic Scholar、OpenAlex 搜索相关论文，支持注入自定义检索源（PubMed、IEEE Xplore、用户文献库等）
- **选题推荐**：基于文献趋势推荐具体可行的论文选题
- **大纲规划**：基于检索结果生成结构化论文大纲，并为每个章节标注建议图表
- **章节撰写**：结合 RAG 上下文自动撰写论文章节，在正文中插入图表占位符（含 chunk_id 绑定）
- **审核修订**：AI 审核机制，支持多轮修改
- **一致性检查**：检查各章节之间的术语、引用、逻辑一致性
- **参考文献生成**：自动调用 CrossRef API 验证 DOI、补全发表年份与期刊/会议信息，生成准确的标准格式参考文献列表；API 不可用时自动回退到本地提取
- **流式调用**：所有节点支持 `astream()` 方法，可实时输出 token
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

### Agent 包内部配置

`seele-scholar-agent` 只管理自身运行所需的极少数配置，通过 `src/seele_scholar_agent/.env` 加载：

```bash
cp src/seele_scholar_agent/.env.example src/seele_scholar_agent/.env
```

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API 密钥（可选，提升频率限制） | 空 |
| `MAX_REVISIONS` | 最大修订轮次 | `3` |

### 调用方配置（由你的项目管理）

LLM、向量数据库等配置由**调用方**自行管理，在初始化时注入到 agent：

```env
# 你的项目 .env（示例）
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1
```

支持任何 OpenAI 兼容 API，通过 `ChatOpenAI` 构造参数传入：

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

### 快速开始（完整工作流）

```python
import asyncio
from datetime import datetime
from uuid import uuid4

from langchain_openai import ChatOpenAI
from seele_scholar_agent.config import settings
from seele_scholar_agent.graph import create_writing_graph
from seele_scholar_agent.state import AgentState
from seele_scholar_agent.agent_config import PromptsConfig


async def main():
    model = ChatOpenAI(
        model="gpt-4o",
        api_key="sk-...",
        base_url="https://api.openai.com/v1",
        temperature=0.7,
    )

    prompts = PromptsConfig(
        planner_system_prompt="你是大纲规划师...",
        planner_user_prompt="研究主题：{topic}\n...",
        writer_system_prompt="你是学术论文撰写者...",
        writer_user_prompt="论文主题：{topic}\n...",
        reviewer_system_prompt="你是审稿人...",
        reviewer_user_prompt="请审阅以下章节：{content}\n...",
        topic_proposer_system_prompt="你是学术导师...",
        topic_proposer_user_prompt="宽泛研究方向：{topic}\n...",
        finalizer_system_prompt="你是摘要撰写者...",
        finalizer_user_prompt="论文主题：{topic}\n...",
        consistency_check_system_prompt="你负责一致性检查...",
        consistency_check_user_prompt="请检查章节一致性：{sections_summary}\n...",
        citation_alignment_system_prompt="你负责引用核查...",
        citation_alignment_user_prompt="请检查引用：{content}\n...",
        topic_translation_system_prompt="You are an academic search query expert...",
        topic_translation_user_prompt="Translate the following topic: {topic}",
    )

    app = create_writing_graph(model=model, prompts=prompts, rag_retriever=None)

    initial_state: AgentState = {
        "thread_id": str(uuid4()),
        "topic": "你的研究主题",
        "language": "zh",
        "created_at": datetime.now(),
        "tenant_id": None,
        "broad_papers": [],
        "proposed_topics": [],
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
        "error_message": None,
        "max_revisions": settings.MAX_REVISIONS,
        "revision_count": 0,
        "references": [],
        "consistency_issues": [],
        "consistency_checked": False,
    }

    result = await app.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": initial_state["thread_id"]}}
    )

    print(f"状态: {result.get('status')}")
    if result.get("outline"):
        print(f"标题: {result['outline'].title}")
        for s in result["outline"].sections:
            print(f"  {s.order}. {s.title}")
            for fig in s.suggested_figures:
                print(f"    [图表] {fig}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 结合 RAG 使用（WriterNode 文献注入）

RAG 上下文通过 `RAGRetrieverFunc` 回调注入，由调用方实现检索逻辑（Qdrant、Chroma 等均可）：

```python
from seele_scholar_agent import RAGRetrieverFunc
from seele_scholar_agent.state import DocumentChunk

# RAGRetrieverFunc = Callable[[str], Awaitable[list[DocumentChunk]]]
# 调用方实现，根据查询返回文档块列表

async def my_rag_retriever(query: str) -> list[DocumentChunk]:
    # 使用 Qdrant、Chroma 等向量数据库检索
    results = await qdrant_store.asimilarity_search(query, k=5)
    return [
        DocumentChunk(
            chunk_id=doc.metadata["chunk_id"],
            content=doc.page_content,
            source=doc.metadata.get("source", ""),
        )
        for doc in results
    ]

app = create_writing_graph(
    model=model,
    prompts=prompts,
    rag_retriever=my_rag_retriever,
)
```

### 注入自定义论文检索源（PaperSearchFunc）

通过 `extra_paper_retrievers` 参数可以注入任意外部论文检索源（PubMed、IEEE Xplore、用户私有文献库等），结果与内置三源（ArXiv、Semantic Scholar、OpenAlex）自动合并去重、按相关度排序：

```python
from seele_scholar_agent import PaperSearchFunc
from seele_scholar_agent.state import PaperMetadata

# PaperSearchFunc = Callable[[str], Awaitable[list[PaperMetadata]]]

async def pubmed_retriever(query: str) -> list[PaperMetadata]:
    """从 PubMed 检索论文"""
    results = await fetch_pubmed(query)
    return [
        PaperMetadata(
            paper_id=f"pubmed:{r['pmid']}",
            title=r["title"],
            authors=r["authors"],
            abstract=r["abstract"],
            url=f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/",
            relevance_score=0.0,   # ResearcherNode 会重新计算排序
            source="user_library",  # 建议使用 "user_library" 标记外部来源
        )
        for r in results
    ]

async def my_library_retriever(query: str) -> list[PaperMetadata]:
    """从私有文献库检索"""
    ...

app = create_writing_graph(
    model=model,
    prompts=prompts,
    rag_retriever=my_rag_retriever,
    extra_paper_retrievers=[pubmed_retriever, my_library_retriever],
)
```

### 断点与恢复执行

`create_writing_graph` 在 `topic_proposer` 和 `planner` 节点后设置了断点，分别等待用户选择选题和确认大纲。

**人工确认模式：**

```python
thread_id = str(uuid4())
config = {"configurable": {"thread_id": thread_id}}

# 第一次调用，运行到 topic_proposer 断点
result = await app.ainvoke(initial_state, config=config)

# 用户选择选题
if result["status"] == "waiting_human":
    for t in result["proposed_topics"]:
        print(f"- {t.title}（{t.difficulty_level}）")
    chosen_topic = result["proposed_topics"][0].title

    # 更新选题并继续到 planner 断点
    app.update_state(config, {"topic": chosen_topic})
    result = await app.ainvoke(None, config=config)

# 用户确认大纲
if result["status"] == "waiting_human":
    outline = result["outline"]
    print(f"标题: {outline.title}")
    for s in outline.sections:
        print(f"  {s.order}. {s.title} — 建议图表: {s.suggested_figures}")

    app.update_state(config, {"outline_approved": True})
    result = await app.ainvoke(None, config=config)
```

**自动模式（跳过断点，适合测试）：**

```python
# 使用 create_simple_writing_graph，无断点
from seele_scholar_agent.graph import create_simple_writing_graph

app = create_simple_writing_graph(model=model, prompts=prompts, rag_retriever=None)
result = await app.ainvoke(initial_state, config=config)
```

### 单节点流式调用

所有节点均支持独立实例化和流式调用，适用于只需要某个功能的场景。

#### 流式事件结构（NodeStreamEvent）

```python
# NodeStreamEvent 是 TypedDict（total=False），type 字段区分事件类型：
# - "token"    : LLM 输出的文本片段，携带 token: str
# - "progress" : 阶段进度提示，携带 progress: str
# - "result"   : 最终结果，携带 result: dict，始终是最后一个事件

async for event in node.astream(state):
    if event["type"] == "token":
        print(event["token"], end="", flush=True)
    elif event["type"] == "progress":
        print(f"\n[{event['progress']}]")
    elif event["type"] == "result":
        data = event.get("result", {})  # 注意：total=False，用 .get() 访问
```

#### 论文检索（ResearcherNode）

```python
from seele_scholar_agent import ResearcherNode

researcher = ResearcherNode(
    llm=model,
    extra_paper_retrievers=[pubmed_retriever],  # 可选
)

async for event in researcher.astream(state):
    if event["type"] == "progress":
        print(f"[{event['progress']}]")
    elif event["type"] == "result":
        papers = event.get("result", {}).get("papers", [])
        for p in papers:
            print(f"[{p.source}] {p.title}")
```

#### 选题推荐（TopicProposerNode）

```python
from seele_scholar_agent import TopicProposerNode

proposer = TopicProposerNode(llm=model, prompts=prompts)

async for event in proposer.astream(state):
    if event["type"] == "result":
        for topic in event.get("result", {}).get("proposed_topics", []):
            print(f"- {topic.title}（{topic.difficulty_level}）")
```

#### 大纲规划（PlannerNode）

```python
from seele_scholar_agent import PlannerNode

planner = PlannerNode(llm=model, prompts=prompts)

async for event in planner.astream(state):
    if event["type"] == "token":
        print(event["token"], end="", flush=True)
    elif event["type"] == "result":
        outline = event.get("result", {}).get("outline")
        for section in outline.sections:
            print(f"{section.order}. {section.title}")
            for fig in section.suggested_figures:
                print(f"   [建议图表] {fig}")
```

#### 章节撰写（WriterNode）

```python
from seele_scholar_agent import WriterNode

writer = WriterNode(llm=model, prompts=prompts, rag_retriever=my_rag_retriever)

async for event in writer.astream(state):
    if event["type"] == "token":
        print(event["token"], end="", flush=True)
    elif event["type"] == "result":
        sections = event.get("result", {}).get("sections", [])
        print(sections[state["current_section_index"]].content)
```

#### 审稿（ReviewerNode）

```python
from seele_scholar_agent import ReviewerNode

reviewer = ReviewerNode(llm=model, prompts=prompts)

async for event in reviewer.astream(state):
    if event["type"] == "result":
        review = event.get("result", {}).get("current_review")
        print(f"通过: {review.approved}, 评分: {review.score}")
        for issue in review.issues:
            print(f"  [{issue.type}] {issue.description}")
```

### 图表占位符处理

Writer 节点在正文中插入图表占位符，格式如下：

```
{{FIGURE: 条形图，展示各模型在ImageNet上的Top-1准确率对比 | chunks:[abc123,def456]}}
{{TABLE: 各方法时间复杂度与空间复杂度对比 | chunks:[xyz789]}}
```

- `FIGURE` / `TABLE`：图表类型
- 描述部分：图表内容和展示目的
- `chunks`：对应 RAG 中数据来源的 `chunk_id` 列表，由 LLM 在写作时从 RAG context 中自动绑定；无数据支撑时为空数组 `chunks:[]`

**主项目解析示例：**

```python
import re

FIGURE_PATTERN = re.compile(
    r'\{\{(FIGURE|TABLE): (.+?) \| chunks:\[([^\]]*)\]\}\}'
)

def extract_figures(content: str):
    results = []
    for fig_type, description, chunks_str in FIGURE_PATTERN.findall(content):
        chunk_ids = [c.strip() for c in chunks_str.split(",") if c.strip()]
        results.append({
            "type": fig_type,
            "description": description,
            "chunk_ids": chunk_ids,
        })
    return results

async def render_figures(content: str, rag_store) -> str:
    for fig_type, description, chunks_str in FIGURE_PATTERN.findall(content):
        chunk_ids = [c.strip() for c in chunks_str.split(",") if c.strip()]
        data = await rag_store.get_chunks(chunk_ids)
        placeholder = f"{{{{{fig_type}: {description} | chunks:[{chunks_str}]}}}}"
        content = content.replace(placeholder, render_chart(fig_type, description, data))
    return content
```

## 节点返回值（result 事件字段）

每个节点 `astream()` 的最后一个事件类型为 `"result"`，`event.get("result", {})` 是一个 dict，对应 `AgentState` 的局部更新。

### TopicProposerNode

| 字段 | 类型 | 说明 |
|------|------|------|
| `broad_papers` | `list[PaperMetadata]` | 宽泛检索阶段找到的文献 |
| `proposed_topics` | `list[ProposedTopic]` | 推荐的论文选题列表 |
| `status` | `"waiting_human"` | 等待用户选择选题 |

`ProposedTopic` 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `title` | `str` | 选题标题 |
| `description` | `str` | 详细描述和切入点 |
| `trend_analysis` | `str` | 趋势分析（受哪些文献启发） |
| `difficulty_level` | `"easy" \| "medium" \| "hard"` | 难度评估 |

### ResearcherNode

| 字段 | 类型 | 说明 |
|------|------|------|
| `papers` | `list[PaperMetadata]` | 检索到的论文列表（多源去重，按相关度排序） |
| `search_queries` | `list[str]` | 实际使用的搜索词列表 |
| `status` | `"planning"` | 进入规划阶段 |

`PaperMetadata` 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `paper_id` | `str` | 论文唯一 ID |
| `title` | `str` | 论文标题 |
| `authors` | `list[str]` | 作者列表 |
| `abstract` | `str` | 摘要 |
| `url` | `str \| None` | 论文页面链接 |
| `pdf_url` | `str \| None` | PDF 直链 |
| `relevance_score` | `float` | 相关度分数 |
| `source` | `"arxiv" \| "semantic_scholar" \| "openalex" \| "user_library"` | 来源 |

### PlannerNode

| 字段 | 类型 | 说明 |
|------|------|------|
| `outline` | `OutlineStructure` | 生成的论文大纲 |
| `sections` | `list[SectionDraft]` | 按大纲拆分的章节草稿（初始均为 pending） |
| `current_section_index` | `0` | 重置章节索引 |
| `status` | `"waiting_human"` | 等待用户确认大纲 |

`SectionOutline` 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `title` | `str` | 章节标题 |
| `description` | `str` | 章节描述 |
| `order` | `int` | 章节顺序编号 |
| `key_points` | `list[str]` | 关键论点列表 |
| `suggested_figures` | `list[str]` | 建议在本章插入的图表描述列表 |

### WriterNode

| 字段 | 类型 | 说明 |
|------|------|------|
| `sections` | `list[SectionDraft]` | 当前章节 content 已填充，status 变为 `"review"` |
| `status` | `"reviewing"` | 进入审稿阶段 |

`SectionDraft` 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `section_id` | `str` | 章节 ID（如 `"section_0"`） |
| `title` | `str` | 章节标题 |
| `content` | `str` | 正文内容（Markdown，含图表占位符） |
| `order_index` | `int` | 章节顺序 |
| `status` | `"pending" \| "writing" \| "review" \| "approved" \| "auto_generated"` | 章节状态 |
| `revision_count` | `int` | 已修订次数 |
| `review_comments` | `list[str]` | 审稿意见（修订时注入） |

### ReviewerNode

审核通过时：

| 字段 | 类型 | 说明 |
|------|------|------|
| `sections` | `list[SectionDraft]` | 当前章节 status 更新为 `"approved"` |
| `sections_completed` | `list[str]` | 新增当前章节标题 |
| `review_history` | `list[dict]` | 新增本次审核记录 |
| `current_review` | `dict` | 本次审核结果（见下表） |
| `current_section_index` | `int` | 推进到下一章节 |
| `status` | `"writing" \| "completed"` | 继续或完成 |

审核不通过时：

| 字段 | 类型 | 说明 |
|------|------|------|
| `sections` | `list[SectionDraft]` | 当前章节 status 变回 `"writing"`，追加审稿意见 |
| `revision_count` | `int` | 修订计数加 1 |
| `status` | `"writing"` | 返回撰写阶段重写 |

`current_review` dict 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `approved` | `bool` | 是否通过 |
| `score` | `int` | 评分（1-10） |
| `issues` | `list[dict]` | 问题列表（含 type、description、suggestion、location） |
| `summary` | `str` | 总体审阅意见 |

`issues[].type` 可选值：`factual_error` \| `missing_citation` \| `weak_argument` \| `format_issue` \| `citation_mismatch` \| `other`

### FinalizerNode

| 字段 | 类型 | 说明 |
|------|------|------|
| `sections` | `list[SectionDraft]`（可选） | 更新了摘要/结论内容（status 为 `"auto_generated"`） |
| `status` | `"completed"` | 工作流完成 |

### ConsistencyCheckerNode

| 字段 | 类型 | 说明 |
|------|------|------|
| `consistency_checked` | `True` | 标记检查已完成 |
| `consistency_issues` | `list[ConsistencyIssue]` | 发现的一致性问题（无问题时为空数组） |

`ConsistencyIssue` 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `issue_type` | `"terminology" \| "citation" \| "logic" \| "other"` | 问题类型 |
| `description` | `str` | 问题描述 |
| `sections_involved` | `list[str]` | 涉及的章节标题列表 |
| `suggestion` | `str` | 修改建议 |

### ReferenceGeneratorNode

| 字段 | 类型 | 说明 |
|------|------|------|
| `references` | `list[ReferenceEntry]` | 参考文献列表（按正文引用顺序排列） |
| `status` | `"completed"` | 工作流完成 |

`ReferenceEntry` 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `number` | `int` | 引用编号，对应正文中的 `[N]` |
| `paper_id` | `str` | 论文 ID |
| `title` | `str` | 论文标题 |
| `authors` | `list[str]` | 作者列表 |
| `year` | `int \| None` | 发表年份 |
| `venue` | `str \| None` | 发表期刊/会议（由 CrossRef 补全） |
| `url` | `str \| None` | 论文链接 |
| `doi` | `str \| None` | DOI 标识符 |
| `formatted` | `str` | 格式化字符串（如 `[1] Author. Title. Venue. (Year)`） |

## 自定义 Prompts（PromptsConfig）

所有节点的 prompt 均通过 `PromptsConfig` 注入，支持完全自定义：

```python
from seele_scholar_agent.agent_config import PromptsConfig

prompts = PromptsConfig(
    planner_system_prompt="你是大纲规划师...",
    planner_user_prompt="研究主题：{topic}\n...",
    writer_system_prompt="你是学术论文撰写者...",
    writer_user_prompt="论文主题：{topic}\n...",
    reviewer_system_prompt="你是审稿人...",
    reviewer_user_prompt="请审阅以下章节：{content}\n...",
    topic_proposer_system_prompt="你是学术导师...",
    topic_proposer_user_prompt="宽泛研究方向：{topic}\n...",
    finalizer_system_prompt="你是摘要撰写者...",
    finalizer_user_prompt="论文主题：{topic}\n...",
    consistency_check_system_prompt="你负责一致性检查...",
    consistency_check_user_prompt="请检查章节一致性：{sections_summary}\n...",
    citation_alignment_system_prompt="你负责引用核查...",
    citation_alignment_user_prompt="请检查引用：{content}\n...",
    topic_translation_system_prompt="You are an academic search query expert...",
    topic_translation_user_prompt="Translate the following topic: {topic}",
)
```

## 项目结构

```
src/seele_scholar_agent/
├── __init__.py             # 公共 API 导出
├── config.py               # 配置管理
├── state.py                # Pydantic 模型和 TypedDict 状态定义
├── graph.py                # LangGraph 工作流定义
├── agent_config.py         # PromptsConfig、RAGRetrieverFunc、PaperSearchFunc
├── logging.py              # 结构化日志配置
├── i18n.py                 # 多语言支持
├── tools/
│   └── crossref.py         # CrossRef REST API 查询（DOI 验证、元数据补全）
└── nodes/
    ├── __init__.py         # NodeStreamEvent、_stream_llm_text、invoke_with_retry
    ├── topic_proposer.py   # 选题推荐节点
    ├── researcher.py       # 论文检索节点（ArXiv、Semantic Scholar、OpenAlex + 自定义源）
    ├── planner.py          # 大纲规划节点（含 suggested_figures）
    ├── writer.py           # 章节撰写节点（含图表占位符插入）
    ├── reviewer.py         # 审稿节点
    ├── finalizer.py        # 摘要/结论生成节点
    ├── consistency_checker.py  # 一致性检查节点
    ├── reference_generator.py  # 参考文献生成节点（含 CrossRef 集成）
    └── (prompts 已移除，通过 PromptsConfig 由调用方注入)
```

## AgentState 状态字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `thread_id` | `str` | 线程 ID，用于对话持久化 |
| `topic` | `str` | 研究主题 |
| `language` | `Literal["zh","en","ja"]` | 输出语言，默认 `"zh"` |
| `created_at` | `datetime` | 创建时间 |
| `tenant_id` | `str \| None` | 租户 ID（多租户场景） |
| `broad_papers` | `list[PaperMetadata]` | 选题阶段检索到的宽泛文献 |
| `proposed_topics` | `list[ProposedTopic]` | 推荐的论文选题列表 |
| `papers` | `list[PaperMetadata]` | 正式写作阶段检索到的论文 |
| `search_queries` | `list[str]` | 搜索查询记录 |
| `outline` | `OutlineStructure \| None` | 生成的大纲（含各章节 `suggested_figures`） |
| `outline_approved` | `bool` | 大纲是否已审核通过 |
| `sections` | `list[SectionDraft]` | 拆分的章节列表 |
| `current_section_index` | `int` | 当前正在撰写的章节索引 |
| `sections_completed` | `list[str]` | 已完成的章节标题列表 |
| `review_history` | `list[dict]` | 审核历史记录 |
| `current_review` | `ReviewResult \| None` | 当前审核结果 |
| `rag_context` | `list[DocumentChunk]` | RAG 检索到的上下文 |
| `references` | `list[ReferenceEntry]` | 生成的参考文献列表 |
| `consistency_issues` | `list[ConsistencyIssue]` | 一致性检查发现的问题 |
| `consistency_checked` | `bool` | 是否已完成一致性检查 |
| `status` | `Literal[...]` | 当前工作流状态 |
| `error_message` | `str \| None` | 错误信息 |
| `max_revisions` | `int` | 最大修订次数 |
| `revision_count` | `int` | 当前修订计数 |

**status 可选值：** `idle` \| `researching` \| `planning` \| `writing` \| `reviewing` \| `finalizing` \| `checking_consistency` \| `waiting_human` \| `completed` \| `failed`

## 工作流程

```
START → topic_proposer → [选题确认] → researcher → planner → [大纲确认] → writer → reviewer
                                                                                      ↓
                                                                                 [审核通过?]
                                                                                      ↓
                                                                           writer(下一节) 或 finalizer
                                                                                      ↓
                                                                           consistency_checker → reference_generator → END
```

1. **TopicProposer**：基于宽泛研究方向推荐 3 个具体选题（断点，等待用户选择）
2. **Researcher**：从 ArXiv、Semantic Scholar、OpenAlex 及自定义检索源检索相关论文
3. **Planner**：生成结构化大纲，每章节附 `suggested_figures`（断点，等待用户确认）
4. **Writer**：根据大纲和 RAG 上下文撰写章节，正文中插入图表占位符（绑定 chunk_id）
5. **Reviewer**：审核章节质量，支持多轮修订
6. **Finalizer**：生成摘要和结论
7. **ConsistencyChecker**：检查全文一致性
8. **ReferenceGenerator**：生成参考文献列表

> `create_simple_writing_graph` 构建无断点版工作流，适合全自动运行场景。

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
