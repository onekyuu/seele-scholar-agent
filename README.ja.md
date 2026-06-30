# seele-scholar-agent

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

[![PyPI version](https://img.shields.io/pypi/v/seele-scholar-agent.svg)](https://pypi.org/project/seele-scholar-agent/)
[![Python version](https://img.shields.io/pypi/pyversions/seele-scholar-agent.svg)](https://pypi.org/project/seele-scholar-agent/)
[![CI](https://github.com/onekyuu/seele-scholar-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/onekyuu/seele-scholar-agent/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/onekyuu/seele-scholar-agent.svg)](https://github.com/onekyuu/seele-scholar-agent/blob/main/LICENSE)

LangGraph ベースの学術文書執筆 agent です。トピック提案、文献検索、アウトライン作成、章の執筆、レビューと修正、一貫性チェック、参考文献生成、既存草稿の継続、document profile に応じた生成フローをサポートします。

## Features

- OpenAlex、Semantic Scholar、ArXiv、およびカスタム検索器から論文を検索。
- 広い研究テーマから具体的な論文トピックを提案。
- 論文タイプを考慮したアウトラインを生成し、章の目的、接続、証拠マップ、図表案を付与。
- 番号付き引用、evidence packet、claim-evidence binding を使って章を執筆。
- 呼び出し側 state から document profile を選択し、日本の大学院出願向け research proposal profile に対応。
- `GraphConfig` でグラフトポロジーを設定し、全文/単一章生成、トピック/アウトライン承認、finalizer/reference/integrity ステップ、草稿統合、exemplar、類似度チェックを切り替え。
- `WritingPolicy` でレビュー動作を制御し、最大修正回数、インライン引用要件、claim-evidence の厳格度、最大回数到達時のフォールバックを指定。
- `BudgetPolicy`、`BudgetState`、length gate、任意の budget allocator で章ごとの分量を制御。
- 構造化された `ExistingContentRef` により、既存草稿の継続、拡張、書き直し、推敲、参照用途を扱う。
- 承認済み exemplar を構成/文体の参考として使い、引用チェーンには追加しません。
- 章をレビューし、草稿を修正し、章ごとに最大修正回数を制限。失敗したレビューを強制承認しません。
- 検索結果を安定した `CitationSource` に正規化してから、執筆と参考文献生成に使用。
- CrossRef / OpenAlex のメタデータ補完を使って参考文献を生成し、本文に引用がない場合は品質問題を返します。
- アウトライン品質、引用の妥当性、claim の根拠、方法論・統計、段落品質、用語・論理・引用の一貫性をチェック。
- 中国語、日本語、英語向けの locale-aware academic style pack を適用。
- `astream()` によるストリーミング出力で UI 統合に対応。

## Install

```bash
git clone https://github.com/onekyuu/seele-scholar-agent
cd seele-scholar-agent

uv sync
```

開発用依存関係：

```bash
uv sync --extra dev
```

## Configuration

このパッケージは agent レベルの設定のみを管理します。LLM の key やモデルは呼び出し側から注入します。

Agent `.env` ファイル：

```bash
cp src/seele_scholar_agent/.env.example src/seele_scholar_agent/.env
```

対応する agent 変数：

| 変数 | 説明 | デフォルト |
| --- | --- | --- |
| `SEMANTIC_SCHOLAR_API_KEY` | 任意の Semantic Scholar API key | 空 |
| `MAX_REVISIONS` | 章ごとの最大レビュー修正回数 | `3` |

呼び出し側の環境変数例：

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

OpenAI 互換エンドポイントは `ChatOpenAI` 経由で利用できます。

## Quality Controls

ワークフローには LLM レビューに加えて、決定的な品質ゲートが含まれます：

- `CitationSourceGateNode`：計画/執筆の前に検索結果を安定した引用可能ソースへ正規化し、診断情報を返します。
- `OutlineQualityGateNode`：purpose、transition、target claims、evidence plan が不足しているアウトラインをブロックし、非実証論文に実証論文型の構成を誤って適用していないか確認します。
- `ReviewerNode`：引用番号、claim-source の対応、方法論・統計上の問題、段落品質、言語別の文体を確認します。
- `LengthGateNode` / `BudgetRevisionNode`：レビュー前に章ごとの分量を検査し、上限付きの分量修正を実行します。
- `PreservationGate`、`CoverageGate`、`ConflictGate`：`existing_content` が指定された場合に、構造化草稿の保持、カバレッジ、衝突を確認します。
- `SimilarityGateNode`：生成文が exemplar chunk に近すぎる場合に警告を返します。
- `ReferenceGeneratorNode`：本文にインライン引用がない場合、参考文献一覧を生成せず `NO_INLINE_CITATIONS` を返します。
- `IntegrityGateNode`：`strict_academic_mode=True` の場合、より厳格な学術的完全性チェックを行います。
- `MaterialRegistry`：指定された場合、アップロード文献、外部検索文献、背景用途のみ、引用不可、信頼済み/低信頼ソースなどの境界を常時チェックします。

RAG コンテキストは evidence packet として扱われ、`chunk_id`、タイトル、著者、年、ページ、章、関連度スコア、関連理由、quote を含みます。Writer の出力は `ClaimEvidenceBinding` によって監査されるため、引用番号の存在だけでなく、その引用が具体的な claim を支えているかを確認できます。

ホストアプリケーションは `status` が `waiting_human`、`failed`、または
`completed` の場合に `quality_issues` と `quality_report` を確認してください。
`waiting_human` は通常の承認待ちだけでなく、呼び出し側の対応が必要な品質ブロックを表すことがあります。

## Caller State Options

呼び出し側は `AgentState` に任意フィールドを渡して、document profile、構成、証拠ポリシー、草稿再利用、exemplar、文体を制御できます：

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
        "writing_locale": "ja-JP",  # zh-CN, ja-JP, en-US, or custom locale
        "style_profile": "thesis",
        "term_glossary": {"LLM": "大規模言語モデル"},
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
        "existing_content": {
            "draft_id": "draft-1",
            "version_id": "v1",
            "segments": [
                {
                    "segment_id": "seg-1",
                    "text": "既存の段落本文。",
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

`material_registry` が渡された場合、ソース境界チェックは常に有効です。ユーザー指定の必須引用文献が本当に関連しているかのチェックは任意で、`check_required_material_relevance=True` の場合のみ実行されます。必須引用文献は `material_registry.entries[].required=True` で指定します。

RAG 検索器は、可能な限り `DocumentChunk.metadata` に `paper_id` または
`source_paper_id`、`title`、`authors`、`year`、`page`、`section`、
`relevance_score`、`why_relevant`、`quote` を入れてください。`paper_id` が
`state["papers"]` の項目と一致すると、インライン引用を具体的な evidence packet に紐付けられます。

## Quick Start

中断なしのワークフローを実行：

```bash
export OPENAI_API_KEY="sk-..."
export SCHOLAR_TOPIC="大規模言語モデルの解釈可能性"
export SCHOLAR_LANGUAGE="ja"

uv run python examples/simple_workflow.py
```

自分のコードで graph を使う：

```python
from langchain_openai import ChatOpenAI

from seele_scholar_agent import BudgetPolicy, GraphConfig, WritingPolicy
from seele_scholar_agent.graph import create_simple_writing_graph
from examples.common import build_initial_state, build_prompts

model = ChatOpenAI(model="gpt-4o-mini", api_key="sk-...")
state = build_initial_state(
    "大規模言語モデルの解釈可能性",
    language="ja",
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

## Examples

実行可能な長めの例は [`examples/`](examples/) にあります：

| ファイル | 用途 |
| --- | --- |
| [`examples/common.py`](examples/common.py) | model、prompt、初期 state の共通 helper |
| [`examples/simple_workflow.py`](examples/simple_workflow.py) | `create_simple_writing_graph` を使う全自動ワークフロー |
| [`examples/full_workflow_with_interrupts.py`](examples/full_workflow_with_interrupts.py) | トピック選択とアウトライン承認を含む人間参加型ワークフロー |
| [`examples/research_proposal_workflow.py`](examples/research_proposal_workflow.py) | research proposal profile と申請書向け policy |
| [`examples/custom_retrievers.py`](examples/custom_retrievers.py) | カスタム RAG と論文検索器の注入 |
| [`examples/stream_nodes.py`](examples/stream_nodes.py) | `astream()` で単一ノードをストリーミング実行 |
| [`examples/figure_placeholders.py`](examples/figure_placeholders.py) | `{{FIGURE: ...}}` と `{{TABLE: ...}}` プレースホルダーの解析 |

リポジトリのルートから任意の例を実行：

```bash
uv run python examples/full_workflow_with_interrupts.py
```

## Core API

このパッケージは 2 つの graph builder と policy/config モデルを公開します：

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

- `create_writing_graph(...)`：トピック提案とアウトライン作成の後に中断します。
- `create_simple_writing_graph(...)`：中断なしで実行し、テストやバッチ処理に向いています。

どちらも以下が必要です：

- `model`：`ChatOpenAI` インスタンス、または互換の LangChain chat model。
- `prompts`：完全な `PromptsConfig`。
- `rag_retriever`：任意の `Callable[[str], Awaitable[list[DocumentChunk]]]`。

任意の制御項目：

- `graph_config`：`GraphConfig` のトポロジースイッチ。`generation_mode=GenerationMode.SINGLE_SECTION`、承認フラグ、草稿統合、exemplar context、similarity gate、後処理ノードなどを指定します。
- `writing_policy`：`WritingPolicy` によるレビュー/引用動作と最大修正回数到達時のフォールバック。
- `budget_policy`：`BudgetPolicy` による length gate の動作。
- `budget_allocator`：任意の動的な章別 budget allocator。
- `extra_paper_retrievers`：追加の非同期論文検索関数。

## Workflow

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

`create_writing_graph()` は `topic_proposer` と `planner` の後で中断するため、呼び出し側はトピックを選択し、アウトラインを承認できます。

## Project Structure

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

## Development

```bash
uv run pytest
uv run ruff check src/
uv run mypy src/
```

パッケージをビルド：

```bash
uv build
```

## License

MIT
