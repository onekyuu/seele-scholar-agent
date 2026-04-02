# seele-scholar-agent

基于 LangGraph 的学术论文写作智能体。自动生成结构化论文大纲、撰写章节、支持人工审核与多轮修订。

## 功能特性

- **研究检索**：从 ArXiv 和 Semantic Scholar 搜索相关论文
- **选题推荐**：基于文献趋势推荐具体可行的论文选题
- **大纲规划**：基于检索结果生成结构化论文大纲，并为每个章节标注建议图表
- **章节撰写**：结合 RAG 上下文自动撰写论文章节，在正文中插入图表占位符（含 chunk_id 绑定）
- **审核修订**：人工审核机制，支持多轮修改
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


async def main():
    model = ChatOpenAI(
        model="gpt-4o",
        api_key="sk-...",
        base_url="https://api.openai.com/v1",
        temperature=0.7,
    )

    app = create_writing_graph(model=model)

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
            if s.suggested_figures:
                for fig in s.suggested_figures:
                    print(f"    [图表] {fig}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 结合 Qdrant 使用（RAG）

```python
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

qdrant = QdrantClient(url="http://localhost:6333", api_key=None)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key="sk-...")

app = create_writing_graph(
    model=model,
    qdrant_client=qdrant,
    embedding_model=embeddings,
)
```

### 单节点流式调用

所有节点均支持独立实例化和流式调用，适用于只需要某个功能（如仅选题、仅审稿）的场景。

#### 流式事件结构（NodeStreamEvent）

```python
from seele_scholar_agent import NodeStreamEvent

# NodeStreamEvent 是 TypedDict，type 字段区分事件类型：
# - "token"    : LLM 输出的文本片段，携带 token: str
# - "progress" : 阶段进度提示，携带 progress: str
# - "result"   : 最终结果，携带 result: dict，始终是最后一个事件
```

#### 选题推荐（TopicProposerNode）

```python
from langchain_openai import ChatOpenAI
from seele_scholar_agent import TopicProposerNode, NodeStreamEvent
from seele_scholar_agent.agent_config import PromptsConfig
from seele_scholar_agent.nodes.prompts import (
    TOPIC_PROPOSER_SYSTEM_PROMPT,
    TOPIC_PROPOSER_USER_PROMPT,
)

model = ChatOpenAI(model="gpt-4o", api_key="sk-...")
prompts = PromptsConfig(
    topic_proposer_system_prompt=TOPIC_PROPOSER_SYSTEM_PROMPT,
    topic_proposer_user_prompt=TOPIC_PROPOSER_USER_PROMPT,
    # 其余字段按需填写
    ...
)

proposer = TopicProposerNode(llm=model, prompts=prompts)

state = {"topic": "大语言模型推理优化", "language": "zh", ...}

async for event in proposer.astream(state):
    if event["type"] == "token":
        print(event["token"], end="", flush=True)
    elif event["type"] == "progress":
        print(f"\n[{event['progress']}]")
    elif event["type"] == "result":
        for topic in event["result"]["proposed_topics"]:
            print(f"- {topic.title}（{topic.difficulty_level}）")
```

#### 大纲规划（PlannerNode）

```python
from seele_scholar_agent import PlannerNode
from seele_scholar_agent.nodes.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT

planner = PlannerNode(llm=model, prompts=prompts)

async for event in planner.astream(state):
    if event["type"] == "token":
        print(event["token"], end="", flush=True)
    elif event["type"] == "result":
        outline = event["result"]["outline"]
        for section in outline.sections:
            print(f"{section.order}. {section.title}")
            # suggested_figures 为该章节建议插入的图表列表
            for fig in section.suggested_figures:
                print(f"   [建议图表] {fig}")
```

#### 章节撰写（WriterNode）

```python
from seele_scholar_agent import WriterNode

writer = WriterNode(llm=model, prompts=prompts)

async for event in writer.astream(state):
    if event["type"] == "token":
        print(event["token"], end="", flush=True)
    elif event["type"] == "result":
        sections = event["result"]["sections"]
        content = sections[state["current_section_index"]].content
        print(content)
```

#### 审稿（ReviewerNode）

```python
from seele_scholar_agent import ReviewerNode

reviewer = ReviewerNode(llm=model, prompts=prompts)

async for event in reviewer.astream(state):
    if event["type"] == "result":
        review = event["result"].get("current_review")
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

# 按 chunk_id 从 RAG 取数据并替换占位符
async def render_figures(content: str, rag_store) -> str:
    for fig_type, description, chunks_str in FIGURE_PATTERN.findall(content):
        chunk_ids = [c.strip() for c in chunks_str.split(",") if c.strip()]
        data = await rag_store.get_chunks(chunk_ids)
        placeholder = f"{{{{{fig_type}: {description} | chunks:[{chunks_str}]}}}}"
        content = content.replace(placeholder, render_chart(fig_type, description, data))
    return content
```

### 节点返回值（result 事件字段）

每个节点 `astream()` 的最后一个事件类型为 `"result"`，`event["result"]` 是一个 dict，对应 `AgentState` 的局部更新。以下是各节点的完整返回字段。

#### TopicProposerNode

| 字段 | 类型 | 说明 |
|------|------|------|
| `broad_papers` | `list[PaperMetadata]` | 宽泛检索阶段找到的文献 |
| `proposed_topics` | `list[ProposedTopic]` | 推荐的论文选题列表 |
| `status` | `"waiting_human"` | 等待用户选择选题 |

`ProposedTopic` 模型字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `title` | `str` | 选题标题 |
| `description` | `str` | 选题详细描述和切入点 |
| `trend_analysis` | `str` | 趋势分析（受哪些文献启发） |
| `difficulty_level` | `"easy" \| "medium" \| "hard"` | 难度评估 |

#### ResearcherNode

| 字段 | 类型 | 说明 |
|------|------|------|
| `papers` | `list[PaperMetadata]` | 检索到的论文列表（多源去重，按相关度排序） |
| `search_queries` | `list[str]` | 实际使用的搜索词列表 |
| `status` | `"planning"` | 进入规划阶段 |

`PaperMetadata` 模型字段：

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

#### PlannerNode

| 字段 | 类型 | 说明 |
|------|------|------|
| `outline` | `OutlineStructure` | 生成的论文大纲 |
| `sections` | `list[SectionDraft]` | 按大纲拆分的章节草稿列表（初始均为 pending） |
| `current_section_index` | `0` | 重置章节索引 |
| `status` | `"waiting_human"` | 等待用户确认大纲 |

`OutlineStructure` 模型字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `title` | `str` | 论文标题 |
| `abstract` | `str` | 摘要（占位，由 Finalizer 生成正式内容） |
| `sections` | `list[SectionOutline]` | 章节大纲列表 |
| `keywords` | `list[str]` | 关键词列表 |

`SectionOutline` 模型字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `title` | `str` | 章节标题 |
| `description` | `str` | 章节描述 |
| `order` | `int` | 章节顺序编号 |
| `key_points` | `list[str]` | 关键论点列表 |
| `suggested_figures` | `list[str]` | 建议在本章插入的图表描述列表 |

#### WriterNode

| 字段 | 类型 | 说明 |
|------|------|------|
| `sections` | `list[SectionDraft]` | 更新后的章节列表（当前章节 content 已填充，status 变为 `"review"`） |
| `status` | `"reviewing"` | 进入审稿阶段 |

`SectionDraft` 模型字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `section_id` | `str` | 章节 ID（如 `"section_0"`） |
| `title` | `str` | 章节标题 |
| `description` | `str` | 章节描述 |
| `content` | `str` | 撰写的正文内容（Markdown 格式，含图表占位符） |
| `order_index` | `int` | 章节顺序 |
| `status` | `"pending" \| "writing" \| "review" \| "approved" \| "auto_generated"` | 章节状态 |
| `revision_count` | `int` | 已修订次数 |
| `review_comments` | `list[str]` | 审稿意见列表（修订时注入） |

`content` 字段中的图表占位符格式：

```
{{FIGURE: 图表描述 | chunks:[chunk_id1,chunk_id2]}}
{{TABLE: 表格描述 | chunks:[chunk_id1]}}
```

#### ReviewerNode

审核通过时：

| 字段 | 类型 | 说明 |
|------|------|------|
| `sections` | `list[SectionDraft]` | 当前章节 status 更新为 `"approved"` |
| `sections_completed` | `list[str]` | 新增当前章节标题 |
| `review_history` | `list[dict]` | 新增本次审核记录（含 section、score、approved、timestamp） |
| `current_review` | `dict` | 本次审核结果（见下表） |
| `current_section_index` | `int` | 推进到下一章节（最后一章时不含此字段） |
| `status` | `"writing" \| "completed"` | 继续写下一章或完成 |

审核不通过时（未超过最大修订次数）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `sections` | `list[SectionDraft]` | 当前章节 status 变回 `"writing"`，review_comments 追加意见 |
| `review_history` | `list[dict]` | 新增本次审核记录 |
| `current_review` | `dict` | 本次审核结果 |
| `revision_count` | `int` | 修订计数加 1 |
| `status` | `"writing"` | 返回撰写阶段重写 |

`current_review` dict 字段（对应 `ReviewResult`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `approved` | `bool` | 是否通过 |
| `score` | `int` | 评分（1-10） |
| `issues` | `list[dict]` | 问题列表（含 type、description、suggestion、location） |
| `summary` | `str` | 总体审阅意见 |

`issues[].type` 可选值：`factual_error` \| `missing_citation` \| `weak_argument` \| `format_issue` \| `citation_mismatch` \| `other`

#### FinalizerNode

| 字段 | 类型 | 说明 |
|------|------|------|
| `sections` | `list[SectionDraft]`（可选） | 更新了摘要/结论章节内容（status 为 `"auto_generated"`），仅当有章节被修改时才包含此字段 |
| `status` | `"completed"` | 工作流完成 |

#### ConsistencyCheckerNode

| 字段 | 类型 | 说明 |
|------|------|------|
| `consistency_checked` | `True` | 标记检查已完成 |
| `consistency_issues` | `list[ConsistencyIssue]` | 发现的一致性问题列表（无问题时为空数组） |

`ConsistencyIssue` 模型字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `issue_type` | `"terminology" \| "citation" \| "logic" \| "other"` | 问题类型 |
| `description` | `str` | 问题描述 |
| `sections_involved` | `list[str]` | 涉及的章节标题列表 |
| `suggestion` | `str` | 修改建议 |

#### ReferenceGeneratorNode

| 字段 | 类型 | 说明 |
|------|------|------|
| `references` | `list[ReferenceEntry]` | 生成的参考文献列表（按正文引用顺序排列） |
| `status` | `"completed"` | 工作流完成 |

`ReferenceEntry` 模型字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `number` | `int` | 引用编号，对应正文中的 `[N]` |
| `paper_id` | `str` | 论文 ID |
| `title` | `str` | 论文标题 |
| `authors` | `list[str]` | 作者列表（CrossRef 有数据时使用 API 返回值） |
| `year` | `int \| None` | 发表年份（优先从 CrossRef 获取） |
| `venue` | `str \| None` | 发表期刊/会议（由 CrossRef 补全，本地无法提取） |
| `url` | `str \| None` | 论文链接 |
| `doi` | `str \| None` | DOI 标识符（从 URL 提取或由 CrossRef 返回） |
| `formatted` | `str` | 格式化的参考文献字符串（如 `[1] Author. Title. Venue. (Year)`） |

---

### 自定义 Prompts（PromptsConfig）

所有节点的 prompt 均通过 `PromptsConfig` 注入，支持完全自定义：

```python
from seele_scholar_agent.agent_config import PromptsConfig
from seele_scholar_agent.nodes.prompts import (
    PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT,
    WRITER_SYSTEM_PROMPT, WRITER_USER_PROMPT,
    REVIEWER_SYSTEM_PROMPT, REVIEWER_USER_PROMPT,
    TOPIC_PROPOSER_SYSTEM_PROMPT, TOPIC_PROPOSER_USER_PROMPT,
    FINALIZER_SYSTEM_PROMPT, FINALIZER_USER_PROMPT,
    CONSISTENCY_CHECK_SYSTEM_PROMPT, CONSISTENCY_CHECK_USER_PROMPT,
    CITATION_ALIGNMENT_SYSTEM_PROMPT, CITATION_ALIGNMENT_USER_PROMPT,
)

# 使用默认 prompts
prompts = PromptsConfig(
    planner_system_prompt=PLANNER_SYSTEM_PROMPT,
    planner_user_prompt=PLANNER_USER_PROMPT,
    writer_system_prompt=WRITER_SYSTEM_PROMPT,
    writer_user_prompt=WRITER_USER_PROMPT,
    reviewer_system_prompt=REVIEWER_SYSTEM_PROMPT,
    reviewer_user_prompt=REVIEWER_USER_PROMPT,
    topic_proposer_system_prompt=TOPIC_PROPOSER_SYSTEM_PROMPT,
    topic_proposer_user_prompt=TOPIC_PROPOSER_USER_PROMPT,
    finalizer_system_prompt=FINALIZER_SYSTEM_PROMPT,
    finalizer_user_prompt=FINALIZER_USER_PROMPT,
    consistency_check_system_prompt=CONSISTENCY_CHECK_SYSTEM_PROMPT,
    consistency_check_user_prompt=CONSISTENCY_CHECK_USER_PROMPT,
    citation_alignment_system_prompt=CITATION_ALIGNMENT_SYSTEM_PROMPT,
    citation_alignment_user_prompt=CITATION_ALIGNMENT_USER_PROMPT,
)
```

## 项目结构

```
seele_scholar_agent/
├── config.py               # 配置管理
├── state.py                # Pydantic 模型和 TypedDict 状态定义
├── graph.py                # LangGraph 工作流定义
├── agent_config.py         # PromptsConfig 和 RAGRetrieverFunc 类型定义
├── logging.py              # 结构化日志配置
├── i18n.py                 # 多语言支持
├── tools/
│   └── crossref.py         # CrossRef REST API 查询（DOI 验证、元数据补全）
└── nodes/
    ├── __init__.py         # NodeStreamEvent、_stream_llm_text、invoke_with_retry
    ├── topic_proposer.py   # 选题推荐节点
    ├── researcher.py       # 论文检索节点（ArXiv、Semantic Scholar）
    ├── planner.py          # 大纲规划节点（含 suggested_figures）
    ├── writer.py           # 章节撰写节点（含图表占位符插入）
    ├── reviewer.py         # 审稿节点
    ├── finalizer.py        # 摘要/结论生成节点
    ├── consistency_checker.py  # 一致性检查节点
    ├── reference_generator.py  # 参考文献生成节点（含 CrossRef 集成）
    └── prompts.py          # 所有节点的默认 LLM 提示词
```

## AgentState 状态字段说明

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
START → researcher → planner → [人工确认] → writer → reviewer
                                                        ↓
                                                   [审核通过?]
                                                        ↓
                                              writer(下一节) 或 finalizer
                                                        ↓
                                              consistency_checker → reference_generator → END
```

1. **TopicProposer**（可选）：基于宽泛研究方向推荐 3 个具体选题
2. **Researcher**：从 ArXiv 和 Semantic Scholar 检索相关论文
3. **Planner**：生成结构化大纲，每章节附 `suggested_figures`（建议图表列表）
4. **Writer**：根据大纲和 RAG 上下文撰写章节，正文中插入图表占位符（绑定 chunk_id）
5. **Reviewer**：审核章节质量，支持多轮修订
6. **Finalizer**：生成摘要和结论
7. **ConsistencyChecker**：检查全文一致性
8. **ReferenceGenerator**：生成参考文献列表

### 断点与恢复执行

graph 在 `planner` 节点后设置了断点（`interrupt_after=["planner"]`），暂停等待人工确认。

**人工确认模式：**

```python
# 第一次调用，运行到断点
result = await app.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})

# 状态变为 waiting_human，等待用户确认大纲
if result["status"] == "waiting_human":
    outline = result["outline"]
    print(f"标题: {outline.title}")
    for s in outline.sections:
        print(f"  {s.order}. {s.title} — 建议图表: {s.suggested_figures}")

    # 用户确认后，更新状态并继续
    app.update_state(config, {"outline_approved": True})
    result = await app.ainvoke(None, config=config)
```

**自动确认模式（测试用）：**

```python
initial_state["outline_approved"] = True
result = await app.ainvoke(initial_state, config=config)
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
    # 创建 LLM（由调用方配置，支持任何 OpenAI 兼容 API）
    model = ChatOpenAI(
        model="gpt-4o",            # 或 "deepseek-chat"、"llama-3.1-70b-versatile" 等
        api_key="sk-...",          # 你的 API Key
        base_url="https://api.openai.com/v1",  # 可替换为其他端点
        temperature=0.7,
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

# 初始化 Qdrant 客户端（由调用方配置）
qdrant = QdrantClient(url="http://localhost:6333", api_key=None)

# 初始化嵌入模型（由调用方配置）
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key="sk-...",
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
