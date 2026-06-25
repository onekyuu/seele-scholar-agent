# ruff: noqa: I001

from collections.abc import Sequence
from typing import Literal

from langchain_openai import ChatOpenAI

from . import _dependency_warnings  # noqa: F401

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from seele_scholar_agent.nodes.topic_proposer import TopicProposerNode

from .agent_config import PaperSearchFunc, PromptsConfig, RAGRetrieverFunc
from .config import settings
from .nodes.consistency_checker import ConsistencyCheckerNode
from .nodes.finalizer import FinalizerNode
from .nodes.integrity_gate import IntegrityGateNode
from .nodes.outline_quality_gate import OutlineQualityGateNode
from .nodes.planner import PlannerNode
from .nodes.reference_generator import ReferenceGeneratorNode
from .nodes.researcher import ResearcherNode
from .nodes.reviewer import ReviewerNode
from .nodes.writer import WriterNode
from .state import AgentState


def create_writing_graph(
    model: ChatOpenAI,
    prompts: PromptsConfig,
    rag_retriever: RAGRetrieverFunc | None,
    semantic_scholar_key: str | None = None,
    openalex_email: str | None = None,
    extra_paper_retrievers: list[PaperSearchFunc] | None = None,
    skip_topic_proposer: bool = False,
    interrupt_after: Sequence[str] | None = ("topic_proposer", "planner"),
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    topic_proposer = TopicProposerNode(llm=model, prompts=prompts)
    researcher = ResearcherNode(
        llm=model,
        prompts=prompts,
        semantic_scholar_key=semantic_scholar_key or settings.SEMANTIC_SCHOLAR_API_KEY,
        openalex_email=openalex_email,
        extra_paper_retrievers=extra_paper_retrievers,
    )
    planner = PlannerNode(llm=model, prompts=prompts)
    outline_quality_gate = OutlineQualityGateNode()
    writer = WriterNode(llm=model, prompts=prompts, rag_retriever=rag_retriever)
    reviewer = ReviewerNode(llm=model, prompts=prompts)
    finalizer = FinalizerNode(llm=model, prompts=prompts)
    consistency_checker = ConsistencyCheckerNode(llm=model, prompts=prompts)
    reference_generator = ReferenceGeneratorNode()
    integrity_gate = IntegrityGateNode()

    graph = StateGraph[AgentState, None, AgentState, AgentState](AgentState)

    graph.add_node("topic_proposer", topic_proposer.propose)
    graph.add_node("researcher", researcher.search)
    graph.add_node("planner", planner.plan)
    graph.add_node("outline_quality_gate", outline_quality_gate.check)
    graph.add_node("writer", writer.write)
    graph.add_node("reviewer", reviewer.review)
    graph.add_node("finalizer", finalizer.finalize)
    graph.add_node("consistency_checker", consistency_checker.check)
    graph.add_node("reference_generator", reference_generator.generate)
    graph.add_node("integrity_gate", integrity_gate.check)

    if skip_topic_proposer:
        graph.add_edge(START, "researcher")
    else:
        graph.add_edge(START, "topic_proposer")
        graph.add_edge("topic_proposer", "researcher")
    graph.add_edge("researcher", "planner")
    graph.add_edge("planner", "outline_quality_gate")
    graph.add_conditional_edges(
        "outline_quality_gate",
        route_quality_gate,
        {"writer": "writer", "end": END},
    )
    graph.add_edge("writer", "reviewer")

    def route_reviewer(state: AgentState) -> Literal["writer", "finalizer", "end"]:
        if state.get("status") == "waiting_human" or _has_blocking_quality_issues(state):
            return "end"

        sections = state["sections"]
        index = state["current_section_index"]

        if sections[index].status == "approved":
            if index + 1 >= len(sections):
                return "finalizer"
            return "writer"
        return "writer"

    graph.add_conditional_edges(
        "reviewer",
        route_reviewer,
        {"writer": "writer", "finalizer": "finalizer", "end": END},
    )
    graph.add_edge("finalizer", "reference_generator")
    graph.add_edge("reference_generator", "consistency_checker")
    graph.add_edge("consistency_checker", "integrity_gate")
    graph.add_edge("integrity_gate", END)

    return _compile_graph(graph, interrupt_after=interrupt_after)


def create_simple_writing_graph(
    model: ChatOpenAI,
    prompts: PromptsConfig,
    rag_retriever: RAGRetrieverFunc | None = None,
    semantic_scholar_key: str | None = None,
    openalex_email: str | None = None,
    extra_paper_retrievers: list[PaperSearchFunc] | None = None,
    skip_topic_proposer: bool = False,
    interrupt_after: Sequence[str] | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    topic_proposer = TopicProposerNode(llm=model, prompts=prompts)
    researcher = ResearcherNode(
        llm=model,
        prompts=prompts,
        semantic_scholar_key=semantic_scholar_key or settings.SEMANTIC_SCHOLAR_API_KEY,
        openalex_email=openalex_email,
        extra_paper_retrievers=extra_paper_retrievers,
    )
    planner = PlannerNode(llm=model, prompts=prompts)
    outline_quality_gate = OutlineQualityGateNode()
    writer = WriterNode(llm=model, prompts=prompts, rag_retriever=rag_retriever)
    reviewer = ReviewerNode(llm=model, prompts=prompts)
    finalizer = FinalizerNode(llm=model, prompts=prompts)
    consistency_checker = ConsistencyCheckerNode(llm=model, prompts=prompts)
    reference_generator = ReferenceGeneratorNode()
    integrity_gate = IntegrityGateNode()

    graph = StateGraph[AgentState, None, AgentState, AgentState](AgentState)

    graph.add_node("proposer", topic_proposer.propose)
    graph.add_node("researcher", researcher.search)
    graph.add_node("planner", planner.plan)
    graph.add_node("outline_quality_gate", outline_quality_gate.check)
    graph.add_node("writer", writer.write)
    graph.add_node("reviewer", reviewer.review)
    graph.add_node("finalizer", finalizer.finalize)
    graph.add_node("consistency_checker", consistency_checker.check)
    graph.add_node("reference_generator", reference_generator.generate)
    graph.add_node("integrity_gate", integrity_gate.check)

    if skip_topic_proposer:
        graph.add_edge(START, "researcher")
    else:
        graph.add_edge(START, "proposer")
        graph.add_edge("proposer", "researcher")
    graph.add_edge("researcher", "planner")
    graph.add_edge("planner", "outline_quality_gate")
    graph.add_conditional_edges(
        "outline_quality_gate",
        route_quality_gate,
        {"writer": "writer", "end": END},
    )
    graph.add_edge("writer", "reviewer")

    def should_continue(state: AgentState) -> Literal["writer", "finalizer", "end"]:
        if state.get("status") == "waiting_human" or _has_blocking_quality_issues(state):
            return "end"

        sections = state["sections"]
        idx = state["current_section_index"]
        if sections[idx].status == "approved" and idx + 1 >= len(sections):
            return "finalizer"
        return "writer"

    graph.add_conditional_edges(
        "reviewer",
        should_continue,
        {"writer": "writer", "finalizer": "finalizer", "end": END},
    )
    graph.add_edge("finalizer", "reference_generator")
    graph.add_edge("reference_generator", "consistency_checker")
    graph.add_edge("consistency_checker", "integrity_gate")
    graph.add_edge("integrity_gate", END)

    return _compile_graph(graph, interrupt_after=interrupt_after)


def _compile_graph(
    graph: StateGraph[AgentState, None, AgentState, AgentState],
    *,
    interrupt_after: Sequence[str] | None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    if interrupt_after is None:
        return graph.compile(checkpointer=MemorySaver())
    return graph.compile(checkpointer=MemorySaver(), interrupt_after=list(interrupt_after))


def _has_blocking_quality_issues(state: AgentState) -> bool:
    return any(
        issue.blocking or issue.severity == "blocking" for issue in state.get("quality_issues", [])
    )


def route_quality_gate(state: AgentState) -> Literal["writer", "end"]:
    if _has_blocking_quality_issues(state):
        return "end"
    return "writer"
