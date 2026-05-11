import asyncio
from os import getenv

from common import build_initial_state, build_model, build_prompts
from seele_scholar_agent.graph import create_simple_writing_graph
from seele_scholar_agent.state import DocumentChunk, PaperMetadata


async def rag_retriever(query: str) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="local-demo-1",
            content=f"Local RAG passage matched query: {query}",
            source="local-demo",
            metadata={"kind": "example"},
        )
    ]


async def paper_retriever(query: str) -> list[PaperMetadata]:
    return [
        PaperMetadata(
            paper_id="local:demo-paper",
            title=f"Local Library Result for {query}",
            authors=["Example Author"],
            abstract="This is a local-library paper returned by a custom retriever.",
            url=None,
            source="user_library",
            relevance_score=0.9,
        )
    ]


async def main() -> None:
    state = build_initial_state(
        topic=getenv("SCHOLAR_TOPIC", "retrieval augmented academic writing"),
        language=getenv("SCHOLAR_LANGUAGE", "zh"),
    )

    app = create_simple_writing_graph(
        model=build_model(),
        prompts=build_prompts(),
        rag_retriever=rag_retriever,
        extra_paper_retrievers=[paper_retriever],
    )

    result = await app.ainvoke(
        state,
        config={"configurable": {"thread_id": state["thread_id"]}},
    )
    print(f"status: {result.get('status')}")
    print(f"papers: {len(result.get('papers', []))}")


if __name__ == "__main__":
    asyncio.run(main())
