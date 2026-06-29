# ruff: noqa: I001

from collections.abc import Sequence
from typing import Literal

from langchain_openai import ChatOpenAI

from . import _dependency_warnings  # noqa: F401

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from seele_scholar_agent.nodes.topic_proposer import TopicProposerNode

from .agent_config import (
    BudgetAllocatorFunc,
    GraphConfig,
    PaperSearchFunc,
    PromptsConfig,
    RAGRetrieverFunc,
)
from .budget import BudgetAllocatorNode, BudgetPolicy, BudgetRevisionNode, LengthGateNode
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
from .policy import SectionExecutionStrategy, WritingPolicy
from .policy.execution_strategy import GenerationMode
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
    graph_config: GraphConfig | None = None,
    writing_policy: WritingPolicy | None = None,
    budget_policy: BudgetPolicy | None = None,
    budget_allocator: BudgetAllocatorFunc | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    graph_config = _resolve_graph_config(graph_config, skip_topic_proposer)
    writing_policy = writing_policy or WritingPolicy()
    budget_policy = budget_policy or BudgetPolicy()
    execution_strategy = SectionExecutionStrategy(graph_config.section_execution_policy())

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
    writer = WriterNode(
        llm=model,
        prompts=prompts,
        rag_retriever=rag_retriever,
        execution_strategy=execution_strategy,
    )
    reviewer = ReviewerNode(
        llm=model,
        prompts=prompts,
        execution_strategy=execution_strategy,
        writing_policy=writing_policy,
    )
    finalizer = FinalizerNode(llm=model, prompts=prompts)
    length_gate = LengthGateNode(budget_policy=budget_policy)
    budget_reviser = BudgetRevisionNode(
        llm=model, prompts=prompts, budget_policy=budget_policy
    )
    budget_allocator_node = BudgetAllocatorNode(budget_allocator)
    consistency_checker = ConsistencyCheckerNode(llm=model, prompts=prompts)
    reference_generator = ReferenceGeneratorNode()
    integrity_gate = IntegrityGateNode()

    graph = StateGraph[AgentState, None, AgentState, AgentState](AgentState)

    graph.add_node("topic_proposer", topic_proposer.propose)
    graph.add_node("researcher", researcher.search)
    graph.add_node("planner", planner.plan)
    graph.add_node("outline_quality_gate", outline_quality_gate.check)
    graph.add_node("writer", writer.write)
    graph.add_node("length_gate", length_gate.check)
    graph.add_node("budget_reviser", budget_reviser.revise)
    graph.add_node("budget_allocator", budget_allocator_node.allocate)
    graph.add_node("reviewer", reviewer.review)
    graph.add_node("finalizer", finalizer.finalize)
    graph.add_node("consistency_checker", consistency_checker.check)
    graph.add_node("reference_generator", reference_generator.generate)
    graph.add_node("integrity_gate", integrity_gate.check)

    if graph_config.generation_mode == GenerationMode.SINGLE_SECTION:
        graph.add_edge(START, "writer")
    elif not graph_config.enable_topic_proposer:
        graph.add_edge(START, "researcher")
    else:
        graph.add_edge(START, "topic_proposer")
        graph.add_edge("topic_proposer", "researcher")
    graph.add_edge("researcher", "planner")
    if graph_config.enable_outline_quality_gate:
        graph.add_edge("planner", "outline_quality_gate")
        graph.add_conditional_edges(
            "outline_quality_gate",
            route_quality_gate,
            {"writer": "writer", "end": END},
        )
    else:
        graph.add_edge("planner", "writer")
    if graph_config.enable_budget_gate and writing_policy.enable_budget_gate:
        graph.add_edge("writer", "length_gate")
        graph.add_conditional_edges(
            "length_gate",
            lambda state: route_length_gate(state, budget_policy),
            {"budget_reviser": "budget_reviser", "reviewer": "reviewer"},
        )
        graph.add_edge("budget_reviser", "length_gate")
    else:
        graph.add_edge("writer", "reviewer")

    completed_route = _first_enabled_postprocess_node(graph_config)

    def route_reviewer(state: AgentState) -> str:
        route = execution_strategy.route_after_review(
            state,
            has_blocking_quality_issues=_has_blocking_quality_issues(state),
            completed_route=completed_route,
        )
        if route == "writer" and budget_allocator is not None:
            return "budget_allocator"
        return route

    graph.add_conditional_edges(
        "reviewer",
        route_reviewer,
        {
            "writer": "writer",
            "budget_allocator": "budget_allocator",
            "finalizer": "finalizer",
            "reference_generator": "reference_generator",
            "consistency_checker": "consistency_checker",
            "integrity_gate": "integrity_gate",
            "end": END,
        },
    )
    graph.add_edge("budget_allocator", "writer")
    _add_postprocess_edges(graph, graph_config)

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
    graph_config: GraphConfig | None = None,
    writing_policy: WritingPolicy | None = None,
    budget_policy: BudgetPolicy | None = None,
    budget_allocator: BudgetAllocatorFunc | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    return create_writing_graph(
        model=model,
        prompts=prompts,
        rag_retriever=rag_retriever,
        semantic_scholar_key=semantic_scholar_key,
        openalex_email=openalex_email,
        extra_paper_retrievers=extra_paper_retrievers,
        skip_topic_proposer=skip_topic_proposer,
        interrupt_after=interrupt_after,
        graph_config=graph_config,
        writing_policy=writing_policy,
        budget_policy=budget_policy,
        budget_allocator=budget_allocator,
    )


def _resolve_graph_config(
    graph_config: GraphConfig | None, skip_topic_proposer: bool
) -> GraphConfig:
    resolved = graph_config or GraphConfig()
    if skip_topic_proposer and resolved.enable_topic_proposer:
        return resolved.model_copy(update={"enable_topic_proposer": False})
    return resolved


def _compile_graph(
    graph: StateGraph[AgentState, None, AgentState, AgentState],
    *,
    interrupt_after: Sequence[str] | None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    if interrupt_after is None:
        return graph.compile(checkpointer=MemorySaver())
    return graph.compile(checkpointer=MemorySaver(), interrupt_after=list(interrupt_after))


def _first_enabled_postprocess_node(graph_config: GraphConfig) -> str:
    if graph_config.enable_finalizer:
        return "finalizer"
    if graph_config.enable_reference_generator:
        return "reference_generator"
    if graph_config.enable_consistency_checker:
        return "consistency_checker"
    if graph_config.enable_integrity_gate:
        return "integrity_gate"
    return "end"


def _next_postprocess_node(graph_config: GraphConfig, current: str) -> str:
    order = [
        ("finalizer", graph_config.enable_finalizer),
        ("reference_generator", graph_config.enable_reference_generator),
        ("consistency_checker", graph_config.enable_consistency_checker),
        ("integrity_gate", graph_config.enable_integrity_gate),
    ]
    seen_current = False
    for node_name, enabled in order:
        if node_name == current:
            seen_current = True
            continue
        if seen_current and enabled:
            return node_name
    return "end"


def _add_postprocess_edges(
    graph: StateGraph[AgentState, None, AgentState, AgentState],
    graph_config: GraphConfig,
) -> None:
    for node_name in (
        "finalizer",
        "reference_generator",
        "consistency_checker",
        "integrity_gate",
    ):
        target = _next_postprocess_node(graph_config, node_name)
        graph.add_edge(node_name, END if target == "end" else target)


def _has_blocking_quality_issues(state: AgentState) -> bool:
    return any(
        issue.blocking or issue.severity == "blocking" for issue in state.get("quality_issues", [])
    )


def route_quality_gate(state: AgentState) -> Literal["writer", "end"]:
    if _has_blocking_quality_issues(state):
        return "end"
    return "writer"


def route_length_gate(
    state: AgentState, budget_policy: BudgetPolicy
) -> Literal["budget_reviser", "reviewer"]:
    diagnostics = state.get("budget_diagnostics") or {}
    if not budget_policy.enabled or not budget_policy.revise_when_over_budget:
        return "reviewer"
    if not diagnostics.get("needs_revision"):
        return "reviewer"

    sections = state.get("sections", [])
    index = state.get("current_section_index", 0)
    if index >= len(sections):
        return "reviewer"
    section_id = sections[index].section_id
    rounds = state.get("budget_revision_rounds", {}) or {}
    if int(rounds.get(section_id, 0)) >= budget_policy.max_budget_revision_rounds:
        return "reviewer"
    return "budget_reviser"
