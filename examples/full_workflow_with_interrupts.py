import asyncio
from os import getenv

from common import build_initial_state, build_model, build_prompts
from seele_scholar_agent.graph import create_writing_graph


async def main() -> None:
    topic = getenv("SCHOLAR_TOPIC", "AI-assisted academic writing")
    language = getenv("SCHOLAR_LANGUAGE", "zh")

    state = build_initial_state(topic=topic, language=language)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    app = create_writing_graph(
        model=build_model(),
        prompts=build_prompts(),
        rag_retriever=None,
    )

    result = await app.ainvoke(state, config=config)

    if result.get("status") == "waiting_human" and result.get("proposed_topics"):
        print("Proposed topics:")
        for idx, proposed in enumerate(result["proposed_topics"], 1):
            print(f"{idx}. {proposed.title} ({proposed.difficulty_level})")

        selected = int(input("Choose a topic number: ").strip() or "1")
        chosen = result["proposed_topics"][selected - 1].title
        app.update_state(config, {"topic": chosen})
        result = await app.ainvoke(None, config=config)

    if result.get("status") == "waiting_human" and result.get("outline"):
        outline = result["outline"]
        print(f"\nOutline: {outline.title}")
        for section in outline.sections:
            print(f"- {section.order}. {section.title}: {section.description}")

        input("\nPress Enter to approve the outline and continue...")
        app.update_state(config, {"outline_approved": True})
        result = await app.ainvoke(None, config=config)

    print(f"\nstatus: {result.get('status')}")
    for section in result.get("sections", []):
        print(f"\n## {section.title}\n{section.content[:500]}")


if __name__ == "__main__":
    asyncio.run(main())
