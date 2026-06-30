# seele-scholar-agent

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

[![PyPI version](https://img.shields.io/pypi/v/seele-scholar-agent.svg)](https://pypi.org/project/seele-scholar-agent/)
[![Python version](https://img.shields.io/pypi/pyversions/seele-scholar-agent.svg)](https://pypi.org/project/seele-scholar-agent/)
[![CI](https://github.com/onekyuu/seele-scholar-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/onekyuu/seele-scholar-agent/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/onekyuu/seele-scholar-agent.svg)](https://github.com/onekyuu/seele-scholar-agent/blob/main/LICENSE)

基于 LangGraph 的学术写作 agent，支持选题推荐、文献检索、大纲规划、章节撰写、审稿修订、一致性检查、参考文献生成、草稿续写和按文档 profile 定制的生成流程。

## 功能特性

- 从 OpenAlex、Semantic Scholar、ArXiv 和自定义检索器搜索论文。
- 从宽泛研究方向推荐具体论文选题。
- 生成感知论文类型的大纲，包含章节目的、过渡、证据映射和建议图表。
- 结合编号引用、evidence packet 和 claim-evidence binding 撰写章节。
- 根据调用方 state 选择 document profile，包括面向日本大学院申请材料的 research proposal profile。
- 使用 `GraphConfig` 配置图拓扑，支持全文/单章节生成、选题/大纲确认、finalizer/reference/integrity 步骤、草稿接入、范例上下文和相似度检查。
- 使用 `WritingPolicy` 控制审稿策略，包括最大修订轮次、内联引用要求、claim-evidence 严格度和达到最大轮次后的兜底行为。
- 使用 `BudgetPolicy`、`BudgetState`、长度 gate 和可选 budget allocator 控制章节预算。
- 通过结构化 `ExistingContentRef` 继续、扩写、重写、润色或参考用户已有草稿。
- 将优秀案例作为结构/风格范例接入，但不会加入引用链。
- 审阅章节、修订草稿，并按章节限制最大修订轮次，不再强制通过未通过审稿的章节。
- 先将检索结果规范化为稳定的 `CitationSource`，再用于写作和参考文献生成。
- 使用 CrossRef / OpenAlex 补全元数据并生成参考文献；正文无内联引用时返回质量问题。
- 检查大纲质量、引用有效性、claim 支撑、方法学与统计、段落质量、术语、逻辑和引用一致性。
- 为中文、日语和英文应用语言感知的学术写作 style pack。
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

## 质量控制

工作流除了 LLM 审稿外，还包含确定性的质量门：

- `CitationSourceGateNode`：在规划/写作前将检索结果规范化为稳定可引用来源，并输出诊断信息。
- `OutlineQualityGateNode`：阻断缺少 purpose、transition、target claims、evidence plan 的大纲，也会检查非实验论文是否误套实验论文结构。
- `ReviewerNode`：检查引用编号、claim-source 支撑、方法学/统计问题、段落质量和语言相关写作风格。
- `LengthGateNode` / `BudgetRevisionNode`：在审稿前执行章节预算检查，并进行有上限的预算返修。
- `PreservationGate`、`CoverageGate` 和 `ConflictGate`：当传入 `existing_content` 时检查结构化草稿的保留、覆盖和冲突。
- `SimilarityGateNode`：当生成文本与范例片段过于相似时返回告警。
- `ReferenceGeneratorNode`：正文没有内联引用时返回 `NO_INLINE_CITATIONS`，不会生成全量参考文献。
- `IntegrityGateNode`：当 `strict_academic_mode=True` 时启用更严格的学术完整性检查。
- `MaterialRegistry`：调用方传入后，常驻检查上传文献、外部检索文献、仅作背景、不可引用、可信/低置信来源等来源边界。

RAG 上下文会升级为 evidence packet，包含 `chunk_id`、标题、作者、年份、页码、章节、相关性分数、相关性说明和 quote。Writer 生成后会通过 `ClaimEvidenceBinding` 审计引用是否支撑具体 claim，而不是只检查引用编号是否存在。

宿主应用应在 `status` 为 `waiting_human`、`failed` 或 `completed` 时检查
`quality_issues` 和 `quality_report`；`waiting_human` 既可能是正常人工确认点，
也可能是需要调用方处理的质量阻断。

## 调用方 State 选项

调用方可以在 `AgentState` 中传入以下可选字段，控制 document profile、结构、证据策略、草稿复用、范例和写作风格：

```python
state.update(
    {
        "document_type": "research_proposal",
        "generation_config": {
            "document_type": "research_proposal",
            "target_chars": 2200,
        },
        "paper_type": "literature_review",
        "structure_pattern": "thematic_review",
        "target_word_count": 6000,
        "strict_academic_mode": True,
        "writing_locale": "zh-CN",  # zh-CN、ja-JP、en-US 或自定义 locale
        "style_profile": "thesis",
        "term_glossary": {"大语言模型": "大型语言模型"},
        "style_pack_override": {
            "display_name": "自定义学位论文风格",
            "general_guidance": ["使用学校指定的学术写作风格。"],
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
        "existing_content": {
            "draft_id": "draft-1",
            "version_id": "v1",
            "segments": [
                {
                    "segment_id": "seg-1",
                    "text": "已有段落正文。",
                    "order": 1,
                    "detected_heading": "Introduction",
                }
            ],
            "preserve_policy": {
                "mode": "preserve_as_much_as_possible",
                "protected_segment_ids": ["seg-1"],
            },
            "user_intent": "expand",
        },
        "exemplar_materials": [
            {
                "exemplar_id": "ex-1",
                "usage_role": "section_reference",
                "outline_patterns": ["Motivation -> gap -> contribution"],
                "style_notes": ["Use cautious synthesis language."],
            }
        ],
        "exemplar_chunks": [
            {
                "exemplar_id": "ex-1",
                "chunk_id": "intro-example",
                "section_title": "Introduction",
                "text": "Example structure/style passage.",
            }
        ],
    }
)
```

只要传入 `material_registry`，来源边界检查就是常驻的。用户指定必引文献的相关性检查是可选功能，只在 `check_required_material_relevance=True` 时运行。必引文献通过 `material_registry.entries[].required=True` 标定。

RAG 检索器应尽量在 `DocumentChunk.metadata` 中提供 `paper_id` 或
`source_paper_id`、`title`、`authors`、`year`、`page`、`section`、
`relevance_score`、`why_relevant` 和 `quote`。如果 `paper_id` 与
`state["papers"]` 中的条目一致，内联引用就可以回连到具体 evidence packet。

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

from seele_scholar_agent import BudgetPolicy, GraphConfig, WritingPolicy
from seele_scholar_agent.graph import create_simple_writing_graph
from examples.common import build_initial_state, build_prompts

model = ChatOpenAI(model="gpt-4o-mini", api_key="sk-...")
state = build_initial_state(
    "大语言模型可解释性研究",
    document_type="academic_paper",
    target_word_count=6000,
)

app = create_simple_writing_graph(
    model=model,
    prompts=build_prompts(),
    rag_retriever=None,
    graph_config=GraphConfig(enable_exemplar_context=False),
    writing_policy=WritingPolicy(max_revisions=3),
    budget_policy=BudgetPolicy(max_budget_revision_rounds=1),
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
| [`examples/research_proposal_workflow.py`](examples/research_proposal_workflow.py) | research proposal profile 与适合研究计划书的 policy |
| [`examples/custom_retrievers.py`](examples/custom_retrievers.py) | 注入自定义 RAG 和论文检索器 |
| [`examples/stream_nodes.py`](examples/stream_nodes.py) | 使用 `astream()` 流式运行单个节点 |
| [`examples/figure_placeholders.py`](examples/figure_placeholders.py) | 解析 `{{FIGURE: ...}}` 和 `{{TABLE: ...}}` 占位符 |

从仓库根目录运行任意示例：

```bash
uv run python examples/full_workflow_with_interrupts.py
```

## 核心 API

该包暴露两个 graph builder 以及 policy/config 模型：

```python
from seele_scholar_agent import (
    BudgetPolicy,
    GenerationMode,
    GraphConfig,
    WritingPolicy,
    create_simple_writing_graph,
    create_writing_graph,
)
```

- `create_writing_graph(...)`：在选题推荐和大纲规划后中断，等待调用方确认。
- `create_simple_writing_graph(...)`：无中断运行，适合测试或批处理。

两者都需要：

- `model`：`ChatOpenAI` 实例或兼容的 LangChain chat model。
- `prompts`：完整的 `PromptsConfig`。
- `rag_retriever`：可选的 `Callable[[str], Awaitable[list[DocumentChunk]]]`。

可选控制项：

- `graph_config`：`GraphConfig` 图拓扑开关，包括 `generation_mode=GenerationMode.SINGLE_SECTION`、人工确认、草稿接入、范例上下文、相似度 gate 和后处理节点。
- `writing_policy`：`WritingPolicy` 审稿/引用行为和最大修订兜底策略。
- `budget_policy`：`BudgetPolicy` 长度 gate 行为。
- `budget_allocator`：可选的动态章节预算分配器。
- `extra_paper_retrievers`：额外异步论文检索函数。

## 工作流

```text
START
  -> topic_proposer
  -> researcher
  -> draft_integration
  -> citation_source_gate
  -> exemplar_planner_context
  -> planner
  -> outline_quality_gate
  -> exemplar_section_retriever
  -> writer
  -> preservation/coverage/conflict gates
  -> similarity_gate
  -> length_gate
  -> budget_reviser
  -> reviewer
  -> finalizer
  -> reference_generator
  -> consistency_checker
  -> integrity_gate
  -> END
```

`create_writing_graph()` 会在 `topic_proposer` 和 `planner` 后中断，调用方可以选择选题并确认大纲。

## 项目结构

```text
src/seele_scholar_agent/
├── agent_config.py
├── document_profile.py
├── config.py
├── graph.py
├── i18n.py
├── logging.py
├── state.py
├── style_packs.py
├── budget/
├── citation/
├── draft/
├── exemplar/
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
├── policy/
├── profiles/
├── review/
├── writing/
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
