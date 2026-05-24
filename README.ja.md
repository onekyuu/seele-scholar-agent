# seele-scholar-agent

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

[![PyPI version](https://img.shields.io/pypi/v/seele-scholar-agent.svg)](https://pypi.org/project/seele-scholar-agent/)
[![Python version](https://img.shields.io/pypi/pyversions/seele-scholar-agent.svg)](https://pypi.org/project/seele-scholar-agent/)
[![CI](https://github.com/onekyuu/seele-scholar-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/onekyuu/seele-scholar-agent/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/onekyuu/seele-scholar-agent.svg)](https://github.com/onekyuu/seele-scholar-agent/blob/main/LICENSE)

LangGraph ベースの学術論文執筆 agent です。トピック提案、文献検索、アウトライン作成、章の執筆、レビューと修正、一貫性チェック、参考文献生成をサポートします。

## Features

- OpenAlex、Semantic Scholar、ArXiv、およびカスタム検索器から論文を検索。
- 広い研究テーマから具体的な論文トピックを提案。
- 論文タイプを考慮したアウトラインを生成し、章の目的、接続、証拠マップ、図表案を付与。
- 番号付き引用、evidence packet、claim-evidence binding を使って章を執筆。
- 章をレビューし、草稿を修正し、章ごとに最大修正回数を制限。失敗したレビューを強制承認しません。
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

- `OutlineQualityGateNode`：purpose、transition、target claims、evidence plan が不足しているアウトラインをブロックし、非実証論文に実証論文型の構成を誤って適用していないか確認します。
- `ReviewerNode`：引用番号、claim-source の対応、方法論・統計上の問題、段落品質、言語別の文体を確認します。
- `ReferenceGeneratorNode`：本文にインライン引用がない場合、参考文献一覧を生成せず `NO_INLINE_CITATIONS` を返します。
- `IntegrityGateNode`：`strict_academic_mode=True` の場合、より厳格な学術的完全性チェックを行います。
- `MaterialRegistry`：指定された場合、アップロード文献、外部検索文献、背景用途のみ、引用不可、信頼済み/低信頼ソースなどの境界を常時チェックします。

RAG コンテキストは evidence packet として扱われ、`chunk_id`、タイトル、著者、年、ページ、章、関連度スコア、関連理由、quote を含みます。Writer の出力は `ClaimEvidenceBinding` によって監査されるため、引用番号の存在だけでなく、その引用が具体的な claim を支えているかを確認できます。

## Caller State Options

呼び出し側は `AgentState` に任意フィールドを渡して、構成、証拠ポリシー、文体を制御できます：

```python
state.update(
    {
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
    }
)
```

`material_registry` が渡された場合、ソース境界チェックは常に有効です。ユーザー指定の必須引用文献が本当に関連しているかのチェックは任意で、`check_required_material_relevance=True` の場合のみ実行されます。必須引用文献は `material_registry.entries[].required=True` で指定します。

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

from seele_scholar_agent.graph import create_simple_writing_graph
from examples.common import build_initial_state, build_prompts

model = ChatOpenAI(model="gpt-4o-mini", api_key="sk-...")
state = build_initial_state("大規模言語モデルの解釈可能性", language="ja")

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

実行可能な長めの例は [`examples/`](examples/) にあります：

| ファイル | 用途 |
| --- | --- |
| [`examples/common.py`](examples/common.py) | model、prompt、初期 state の共通 helper |
| [`examples/simple_workflow.py`](examples/simple_workflow.py) | `create_simple_writing_graph` を使う全自動ワークフロー |
| [`examples/full_workflow_with_interrupts.py`](examples/full_workflow_with_interrupts.py) | トピック選択とアウトライン承認を含む人間参加型ワークフロー |
| [`examples/custom_retrievers.py`](examples/custom_retrievers.py) | カスタム RAG と論文検索器の注入 |
| [`examples/stream_nodes.py`](examples/stream_nodes.py) | `astream()` で単一ノードをストリーミング実行 |
| [`examples/figure_placeholders.py`](examples/figure_placeholders.py) | `{{FIGURE: ...}}` と `{{TABLE: ...}}` プレースホルダーの解析 |

リポジトリのルートから任意の例を実行：

```bash
uv run python examples/full_workflow_with_interrupts.py
```

## Core API

このパッケージは 2 つの graph builder を公開します：

```python
from seele_scholar_agent.graph import create_simple_writing_graph, create_writing_graph
```

- `create_writing_graph(...)`：トピック提案とアウトライン作成の後に中断します。
- `create_simple_writing_graph(...)`：中断なしで実行し、テストやバッチ処理に向いています。

どちらも以下が必要です：

- `model`：`ChatOpenAI` インスタンス、または互換の LangChain chat model。
- `prompts`：完全な `PromptsConfig`。
- `rag_retriever`：任意の `Callable[[str], Awaitable[list[DocumentChunk]]]`。

`extra_paper_retrievers` で追加の論文ソースを注入できます。

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

`create_writing_graph()` は `topic_proposer` と `planner` の後で中断するため、呼び出し側はトピックを選択し、アウトラインを承認できます。

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

パッケージをビルド：

```bash
uv build
```

## License

MIT
