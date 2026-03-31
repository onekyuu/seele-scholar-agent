from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from seele_scholar_agent.nodes.topic_proposer import TopicProposerNode

from .config import settings
from .nodes.planner import PlannerNode
from .nodes.researcher import ResearcherNode
from .nodes.reviewer import ReviewerNode
from .nodes.writer import WriterNode
from .state import AgentState
from .agent_config import PromptsConfig, RAGRetrieverFunc


def create_writing_graph(
    model: ChatOpenAI,
    prompts: PromptsConfig,
    rag_retriever: RAGRetrieverFunc | None,
    semantic_scholar_key: str | None = None,
    openalex_email: str | None = None
) -> CompiledStateGraph:
    topic_proposer = TopicProposerNode(model=model, prompts=prompts)
    researcher = ResearcherNode(
        semantic_scholar_key=semantic_scholar_key or settings.SEMANTIC_SCHOLAR_API_KEY,
        openalex_email=openalex_email
    )
    planner = PlannerNode(model=model, prompts=prompts)

    writer = WriterNode(model=model, prompts=prompts, rag_retriever=rag_retriever)
    reviewer = ReviewerNode(model=model, prompts=prompts)

    graph = StateGraph(AgentState)

    graph.add_node("proposer", topic_proposer.propose)
    graph.add_node("researcher", researcher.search)
    graph.add_node("planner", planner.plan)
    graph.add_node("writer", writer.write)
    graph.add_node("reviewer", reviewer.review)

    graph.add_edge(START, "topic_proposer")
    graph.add_edge("topic_proposer", "researcher")
    graph.add_edge("researcher", "planner")

    graph.add_edge("planner", "writer")
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

    return graph.compile(checkpointer=MemorySaver(), interrupt_after=["topic_proposer", "planner"])


def create_simple_writing_graph(
    model: ChatOpenAI,
    prompts: PromptsConfig,
    rag_retriever: RAGRetrieverFunc | None = None,
    semantic_scholar_key: str | None = None,
    openalex_email: str | None = None
) -> CompiledStateGraph:
    topic_proposer = TopicProposerNode(model=model, prompts=prompts)
    researcher = ResearcherNode(
        semantic_scholar_key=semantic_scholar_key or settings.SEMANTIC_SCHOLAR_API_KEY,
        openalex_email=openalex_email
    )
    planner = PlannerNode(model=model, prompts=prompts)
    writer = WriterNode(model=model, prompts=prompts, rag_retriever=rag_retriever)
    reviewer = ReviewerNode(model=model, prompts=prompts)

    graph = StateGraph(AgentState)

    graph.add_node("proposer", topic_proposer.propose)
    graph.add_node("researcher", researcher.search)
    graph.add_node("planner", planner.plan)
    graph.add_node("writer", writer.write)
    graph.add_node("reviewer", reviewer.review)

    graph.add_edge(START, "proposer")
    graph.add_edge("proposer", "researcher")
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
