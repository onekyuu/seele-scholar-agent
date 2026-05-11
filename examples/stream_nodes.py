import asyncio
from os import getenv

from common import build_initial_state, build_model, build_prompts
from seele_scholar_agent.nodes.planner import PlannerNode
from seele_scholar_agent.state import PaperMetadata


async def main() -> None:
    state = build_initial_state(
        topic=getenv("SCHOLAR_TOPIC", "large language model evaluation"),
        language=getenv("SCHOLAR_LANGUAGE", "zh"),
    )
    state["papers"] = [
        PaperMetadata(
            paper_id="example:1",
            title="Example Paper",
            authors=["Example Author"],
            abstract="A short example abstract.",
            source="user_library",
        )
    ]

    planner = PlannerNode(llm=build_model(), prompts=build_prompts())
    async for event in planner.astream(state):
        event_type = event.get("type")
        if event_type == "token":
            print(event.get("token", ""), end="", flush=True)
        elif event_type == "progress":
            print(f"\n[{event.get('progress')}]\n")
        elif event_type == "result":
            outline = event.get("result", {}).get("outline")
            print(f"\n\noutline title: {outline.title if outline else 'N/A'}")


if __name__ == "__main__":
    asyncio.run(main())
