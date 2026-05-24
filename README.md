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
- Generate paper-type-aware outlines with section purpose, transitions, evidence maps, and suggested figures or tables.
- Draft sections with numbered citations, evidence packets, and claim-evidence bindings.
- Review sections, revise drafts, and cap revisions per section without force-approving failed reviews.
- Generate references with CrossRef/OpenAlex metadata enrichment and reject papers with no inline citations.
- Check outline quality, citation validity, claim support, methodology, paragraph quality, terminology, logic, and citation consistency.
- Apply locale-aware style packs for Chinese, Japanese, and English academic writing.
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

## Quality Controls

The workflow includes deterministic gates in addition to LLM review:

- `OutlineQualityGateNode`: blocks incomplete outlines that lack purpose, transitions, target claims, evidence plans, or use an empirical template for a non-empirical paper type.
- `ReviewerNode`: checks citation numbering, claim-source support, methodology/statistics issues, paragraph quality, and locale-specific writing style.
- `ReferenceGeneratorNode`: returns `NO_INLINE_CITATIONS` instead of generating a full bibliography when the draft has no inline citations.
- `IntegrityGateNode`: enforces strict academic checks when `strict_academic_mode=True`.
- `MaterialRegistry`: always enforces citation boundaries for uploaded, external, background-only, excluded, trusted, normal, or low-confidence materials when provided.

The RAG context is represented as evidence packets with `chunk_id`, title, authors, year, page, section, relevance score, relevance rationale, and quote. Writer output is audited through `ClaimEvidenceBinding` so citations are checked against the claim they support, not only against reference numbers.

## Caller State Options

The caller can pass optional fields in `AgentState` to control structure, evidence policy, and style:

```python
state.update(
    {
        "paper_type": "literature_review",
        "structure_pattern": "thematic_review",
        "target_word_count": 6000,
        "strict_academic_mode": True,
        "writing_locale": "zh-CN",  # zh-CN, ja-JP, en-US, or a custom locale
        "style_profile": "thesis",
        "term_glossary": {"大语言模型": "大型语言模型"},
        "style_pack_override": {
            "display_name": "Custom thesis style",
            "general_guidance": ["Use the institution-specific academic style."],
        },
        "material_registry": {
            "entries": [
                {
                    "paper_id": "user-paper-1",
                    "source_origin": "user_upload",
                    "citation_role": "citable",
                    "confidence": "trusted",
                    "required": True,
                }
            ]
        },
        "check_required_material_relevance": True,
    }
)
```

`material_registry` source-boundary checks are always active when a registry is provided. The required-material relevance check is optional and only runs when `check_required_material_relevance=True`. Required citations are marked with `material_registry.entries[].required=True`.

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
  -> outline_quality_gate
  -> writer <-> reviewer
  -> finalizer
  -> reference_generator
  -> consistency_checker
  -> integrity_gate
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
├── style_packs.py
├── nodes/
│   ├── topic_proposer.py
│   ├── researcher.py
│   ├── planner.py
│   ├── outline_quality_gate.py
│   ├── writer.py
│   ├── reviewer.py
│   ├── finalizer.py
│   ├── reference_generator.py
│   ├── consistency_checker.py
│   ├── integrity_gate.py
│   └── language_style_audit.py
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
