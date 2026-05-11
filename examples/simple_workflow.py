import asyncio
from os import getenv

from common import build_initial_state, build_model, build_prompts
from seele_scholar_agent.graph import create_simple_writing_graph


async def main() -> None:
    topic = getenv("SCHOLAR_TOPIC", "Large language model interpretability")
    language = getenv("SCHOLAR_LANGUAGE", "zh")

    state = build_initial_state(topic=topic, language=language)
    app = create_simple_writing_graph(
        model=build_model(),
        prompts=build_prompts(),
        rag_retriever=None,
    )

    result = await app.ainvoke(
        state,
        config={"configurable": {"thread_id": state["thread_id"]}},
    )

    print(f"status: {result.get('status')}")
    outline = result.get("outline")
    if outline:
        print(f"title: {outline.title}")
    for section in result.get("sections", []):
        print(f"\n## {section.title}\n{section.content[:500]}")


if __name__ == "__main__":
    asyncio.run(main())
