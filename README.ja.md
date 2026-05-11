# seele-scholar-agent

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

LangGraph ベースの学術論文執筆 agent です。トピック提案、文献検索、アウトライン作成、章の執筆、レビューと修正、一貫性チェック、参考文献生成をサポートします。

## Features

- OpenAlex、Semantic Scholar、ArXiv、およびカスタム検索器から論文を検索。
- 広い研究テーマから具体的な論文トピックを提案。
- 図表案付きの構造化アウトラインを生成。
- 番号付き引用と任意の RAG コンテキストを使って章を執筆。
- 章をレビューし、草稿を修正し、章ごとに最大修正回数を制限。
- CrossRef のメタデータ補完を使って参考文献を生成。
- 参考文献生成後に、用語・論理・引用の一貫性をチェック。
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
  -> writer <-> reviewer
  -> finalizer
  -> reference_generator
  -> consistency_checker
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

パッケージをビルド：

```bash
uv build
```

## License

MIT
