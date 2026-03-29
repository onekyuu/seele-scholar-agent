from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .config import settings
from .nodes.planner import PlannerNode
from .nodes.researcher import ResearcherNode
from .nodes.reviewer import ReviewerNode
from .nodes.writer import WriterNode
from .state import AgentState


def create_writing_graph(
    model: ChatOpenAI,
    qdrant_client=None,
    embedding_model=None,
    semantic_scholar_key: str | None = None,
    openalex_email: str | None = None
) -> CompiledStateGraph:
    researcher = ResearcherNode(
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        semantic_scholar_key=semantic_scholar_key or settings.SEMANTIC_SCHOLAR_API_KEY,
        openalex_email=openalex_email
    )
    planner = PlannerNode(model=model)

    writer = WriterNode(model=model)
    reviewer = ReviewerNode(model=model)

    graph = StateGraph(AgentState)

    graph.add_node("researcher", researcher.search)
    graph.add_node("planner", planner.plan)
    graph.add_node("writer", writer.write)
    graph.add_node("reviewer", reviewer.review)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "planner")

    graph.add_edge("planner", END)
    graph.add_edge("writer", "reviewer")

    def route_reviewer(state: AgentState) -> Literal["writer", "__end__"]:
        sections = state["sections"]
        index = state["current_section_index"]

        if sections[index].status == "approved":
            if index + 1 >= len(sections):
                return "__end__"
            return "writer"
        return "writer"

    graph.add_conditional_edges("reviewer", route_reviewer, {"writer": "writer", "__end__": END})

    return graph.compile(checkpointer=MemorySaver(), interrupt_after=["planner"])


def create_simple_writing_graph(
    model: ChatOpenAI,
    qdrant_client=None,
    embedding_model=None,
    semantic_scholar_key: str | None = None,
    openalex_email: str | None = None
) -> CompiledStateGraph:
    researcher = ResearcherNode(
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        semantic_scholar_key=semantic_scholar_key or settings.SEMANTIC_SCHOLAR_API_KEY,
        openalex_email=openalex_email
    )
    planner = PlannerNode(model=model)
    writer = WriterNode(model=model)
    reviewer = ReviewerNode(model=model)

    graph = StateGraph(AgentState)

    graph.add_node("researcher", researcher.search)
    graph.add_node("planner", planner.plan)
    graph.add_node("writer", writer.write)
    graph.add_node("reviewer", reviewer.review)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "planner")
    graph.add_edge("planner", "writer")
    graph.add_edge("writer", "reviewer")

    def should_continue(state: AgentState) -> Literal["writer", "__end__"]:
        sections = state["sections"]
        idx = state["current_section_index"]
        if sections[idx].status == "approved" and idx + 1 >= len(sections):
            return "__end__"
        return "writer"

    graph.add_conditional_edges("reviewer", should_continue, {"writer": "writer", "__end__": END})

    return graph.compile(checkpointer=MemorySaver())
