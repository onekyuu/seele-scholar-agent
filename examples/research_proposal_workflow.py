import asyncio
from os import getenv

from common import build_model, build_prompts, build_research_proposal_state
from seele_scholar_agent import BudgetPolicy, GraphConfig, WritingPolicy
from seele_scholar_agent.graph import create_simple_writing_graph


async def main() -> None:
    state = build_research_proposal_state(
        topic=getenv("SCHOLAR_TOPIC", "LLM-based support for academic writing"),
        language=getenv("SCHOLAR_LANGUAGE", "ja"),
        target_chars=int(getenv("SCHOLAR_TARGET_CHARS", "2200")),
    )

    app = create_simple_writing_graph(
        model=build_model(),
        prompts=build_prompts(),
        rag_retriever=None,
        skip_topic_proposer=True,
        graph_config=GraphConfig(
            require_outline_approval=False,
            enable_exemplar_context=getenv("SCHOLAR_ENABLE_EXEMPLARS") == "1",
            enable_similarity_gate=getenv("SCHOLAR_ENABLE_EXEMPLARS") == "1",
        ),
        writing_policy=WritingPolicy(
            require_inline_citations=False,
            strict_claim_evidence_binding=False,
            allow_uncited_plan_statements=True,
        ),
        budget_policy=BudgetPolicy(max_budget_revision_rounds=1),
    )

    result = await app.ainvoke(
        state,
        config={"configurable": {"thread_id": state["thread_id"]}},
    )

    print(f"status: {result.get('status')}")
    for issue in result.get("quality_issues", []):
        print(f"{issue.severity}: {issue.code} - {issue.message}")
    for section in result.get("sections", []):
        print(f"\n## {section.title}\n{section.content[:500]}")


if __name__ == "__main__":
    asyncio.run(main())
