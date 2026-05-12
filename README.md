# seele-scholar-agent

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

[![PyPI version](https://img.shields.io/pypi/v/seele-scholar-agent.svg)](https://pypi.org/project/seele-scholar-agent/)
[![Python version](https://img.shields.io/pypi/pyversions/seele-scholar-agent.svg)](https://pypi.org/project/seele-scholar-agent/)
[![CI](https://github.com/onekyuu/seele-scholar-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/onekyuu/seele-scholar-agent/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/onekyuu/seele-scholar-agent.svg)](https://github.com/onekyuu/seele-scholar-agent/blob/main/LICENSE)

LangGraph-based academic paper writing agent for topic proposal, literature retrieval, outline planning, section drafting, review loops, consistency checks, and reference generation.

## Features

- Search papers from OpenAlex, Semantic Scholar, ArXiv, and custom retrievers.
- Propose concrete paper topics from broad research directions.
- Generate structured outlines with suggested figures and tables.
- Draft sections with numbered citations and optional RAG context.
- Review sections, revise drafts, and cap revisions per section.
- Generate references with CrossRef metadata enrichment.
- Check terminology, logic, and citation consistency after references are generated.
- Stream node output with `astream()` for UI integration.

## Install

```bash
git clone https://github.com/onekyuu/seele-scholar-agent
cd seele-scholar-agent

uv sync
```

For development dependencies:

```bash
uv sync --extra dev
```

## Configuration

This package only owns agent-level configuration. LLM keys and models are injected by the caller.

Agent `.env` file:

```bash
cp src/seele_scholar_agent/.env.example src/seele_scholar_agent/.env
```

Supported agent variables:

| Variable | Description | Default |
| --- | --- | --- |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional Semantic Scholar API key | empty |
| `MAX_REVISIONS` | Max review cycles per section | `3` |

Example caller environment:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

Any OpenAI-compatible endpoint can be used through `ChatOpenAI`.

## Quick Start

Run the no-interrupt workflow:

```bash
export OPENAI_API_KEY="sk-..."
export SCHOLAR_TOPIC="large language model interpretability"
export SCHOLAR_LANGUAGE="zh"

uv run python examples/simple_workflow.py
```

Use the graph in your own code:

```python
from langchain_openai import ChatOpenAI

from seele_scholar_agent.graph import create_simple_writing_graph
from examples.common import build_initial_state, build_prompts

model = ChatOpenAI(model="gpt-4o-mini", api_key="sk-...")
state = build_initial_state("large language model interpretability")

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

## Examples

Large runnable examples live in [`examples/`](examples/):

| File | Purpose |
| --- | --- |
| [`examples/common.py`](examples/common.py) | Shared model, prompt, and initial-state helpers |
| [`examples/simple_workflow.py`](examples/simple_workflow.py) | Full automatic workflow with `create_simple_writing_graph` |
| [`examples/full_workflow_with_interrupts.py`](examples/full_workflow_with_interrupts.py) | Human-in-the-loop topic and outline approval |
| [`examples/custom_retrievers.py`](examples/custom_retrievers.py) | Inject custom RAG and paper retrievers |
| [`examples/stream_nodes.py`](examples/stream_nodes.py) | Stream a single node with `astream()` |
| [`examples/figure_placeholders.py`](examples/figure_placeholders.py) | Parse `{{FIGURE: ...}}` and `{{TABLE: ...}}` placeholders |

Run any example from the repository root:

```bash
uv run python examples/full_workflow_with_interrupts.py
```

## Core API

The package exposes two graph builders:

```python
from seele_scholar_agent.graph import create_simple_writing_graph, create_writing_graph
```

- `create_writing_graph(...)`: includes interrupts after topic proposal and outline planning.
- `create_simple_writing_graph(...)`: runs without interrupts and is useful for tests or batch jobs.

Both require:

- `model`: a `ChatOpenAI` instance or compatible LangChain chat model.
- `prompts`: a complete `PromptsConfig`.
- `rag_retriever`: optional `Callable[[str], Awaitable[list[DocumentChunk]]]`.

Optional paper sources can be injected with `extra_paper_retrievers`.

## Workflow

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

`create_writing_graph()` interrupts after `topic_proposer` and `planner` so the caller can choose a topic and approve the outline.

## Project Structure

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

## Development

```bash
uv run pytest
uv run ruff check src/
uv run mypy src/
```

Build the package:

```bash
uv build
```

## License

MIT
