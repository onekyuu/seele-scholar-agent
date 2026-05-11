# seele-scholar-agent

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

基于 LangGraph 的学术论文写作 agent，支持选题推荐、文献检索、大纲规划、章节撰写、审稿修订、一致性检查和参考文献生成。

## 功能特性

- 从 OpenAlex、Semantic Scholar、ArXiv 和自定义检索器搜索论文。
- 从宽泛研究方向推荐具体论文选题。
- 生成结构化论文大纲，并为章节标注建议图表。
- 结合编号引用和可选 RAG 上下文撰写章节。
- 审阅章节、修订草稿，并按章节限制最大修订轮次。
- 使用 CrossRef 补全元数据并生成参考文献。
- 在参考文献生成后检查术语、逻辑和引用一致性。
- 使用 `astream()` 流式输出节点结果，便于 UI 集成。

## 安装

```bash
git clone https://github.com/onekyuu/seele-scholar-agent
cd seele-scholar-agent

uv sync
```

安装开发依赖：

```bash
uv sync --extra dev
```

## 配置

该包只管理 agent 级配置。LLM key 和模型由调用方注入。

Agent `.env` 文件：

```bash
cp src/seele_scholar_agent/.env.example src/seele_scholar_agent/.env
```

支持的 agent 变量：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `SEMANTIC_SCHOLAR_API_KEY` | 可选的 Semantic Scholar API key | 空 |
| `MAX_REVISIONS` | 每个章节的最大审稿修订轮次 | `3` |

调用方环境变量示例：

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

任何 OpenAI 兼容端点都可以通过 `ChatOpenAI` 接入。

## 快速开始

运行无断点工作流：

```bash
export OPENAI_API_KEY="sk-..."
export SCHOLAR_TOPIC="大语言模型可解释性研究"
export SCHOLAR_LANGUAGE="zh"

uv run python examples/simple_workflow.py
```

在自己的代码中使用 graph：

```python
from langchain_openai import ChatOpenAI

from seele_scholar_agent.graph import create_simple_writing_graph
from examples.common import build_initial_state, build_prompts

model = ChatOpenAI(model="gpt-4o-mini", api_key="sk-...")
state = build_initial_state("大语言模型可解释性研究")

app = create_simple_writing_graph(
    model=model,
    prompts=build_prompts(),
    rag_retriever=None,
)

result = await app.ainvoke(
    state,
    config={"configurable": {"thread_id": state["thread_id"]}},
)
```

## 示例

较长的可运行示例放在 [`examples/`](examples/)：

| 文件 | 用途 |
| --- | --- |
| [`examples/common.py`](examples/common.py) | 共享 model、prompt 和初始 state helper |
| [`examples/simple_workflow.py`](examples/simple_workflow.py) | 使用 `create_simple_writing_graph` 的全自动工作流 |
| [`examples/full_workflow_with_interrupts.py`](examples/full_workflow_with_interrupts.py) | 带人工选题和大纲确认的工作流 |
| [`examples/custom_retrievers.py`](examples/custom_retrievers.py) | 注入自定义 RAG 和论文检索器 |
| [`examples/stream_nodes.py`](examples/stream_nodes.py) | 使用 `astream()` 流式运行单个节点 |
| [`examples/figure_placeholders.py`](examples/figure_placeholders.py) | 解析 `{{FIGURE: ...}}` 和 `{{TABLE: ...}}` 占位符 |

从仓库根目录运行任意示例：

```bash
uv run python examples/full_workflow_with_interrupts.py
```

## 核心 API

该包暴露两个 graph builder：

```python
from seele_scholar_agent.graph import create_simple_writing_graph, create_writing_graph
```

- `create_writing_graph(...)`：在选题推荐和大纲规划后中断，等待调用方确认。
- `create_simple_writing_graph(...)`：无中断运行，适合测试或批处理。

两者都需要：

- `model`：`ChatOpenAI` 实例或兼容的 LangChain chat model。
- `prompts`：完整的 `PromptsConfig`。
- `rag_retriever`：可选的 `Callable[[str], Awaitable[list[DocumentChunk]]]`。

可通过 `extra_paper_retrievers` 注入额外论文来源。

## 工作流

```text
START
  -> topic_proposer
  -> researcher
  -> planner
  -> writer <-> reviewer
  -> finalizer
  -> reference_generator
  -> consistency_checker
  -> END
```

`create_writing_graph()` 会在 `topic_proposer` 和 `planner` 后中断，调用方可以选择选题并确认大纲。

## 项目结构

```text
src/seele_scholar_agent/
├── agent_config.py
├── config.py
├── graph.py
├── i18n.py
├── logging.py
├── state.py
├── nodes/
│   ├── topic_proposer.py
│   ├── researcher.py
│   ├── planner.py
│   ├── writer.py
│   ├── reviewer.py
│   ├── finalizer.py
│   ├── reference_generator.py
│   └── consistency_checker.py
└── tools/
    └── crossref.py
```

## 开发

```bash
uv run pytest
uv run ruff check src/
uv run mypy src/
```

构建包：

```bash
uv build
```

## 许可证

MIT
