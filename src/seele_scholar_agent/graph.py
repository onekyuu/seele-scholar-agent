from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from seele_scholar_agent.nodes.topic_proposer import TopicProposerNode

from .agent_config import PaperSearchFunc, PromptsConfig, RAGRetrieverFunc
from .config import settings
from .nodes.consistency_checker import ConsistencyCheckerNode
from .nodes.finalizer import FinalizerNode
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
) -> CompiledStateGraph:
    topic_proposer = TopicProposerNode(llm=model, prompts=prompts)
    researcher = ResearcherNode(
        llm=model,
        prompts=prompts,
        semantic_scholar_key=semantic_scholar_key or settings.SEMANTIC_SCHOLAR_API_KEY,
        openalex_email=openalex_email,
        extra_paper_retrievers=extra_paper_retrievers,
    )
    planner = PlannerNode(llm=model, prompts=prompts)
    writer = WriterNode(llm=model, prompts=prompts, rag_retriever=rag_retriever)
    reviewer = ReviewerNode(llm=model, prompts=prompts)
    finalizer = FinalizerNode(llm=model, prompts=prompts)
    consistency_checker = ConsistencyCheckerNode(llm=model, prompts=prompts)
    reference_generator = ReferenceGeneratorNode()

    graph = StateGraph(AgentState)

    graph.add_node("topic_proposer", topic_proposer.propose)
    graph.add_node("researcher", researcher.search)
    graph.add_node("planner", planner.plan)
    graph.add_node("writer", writer.write)
    graph.add_node("reviewer", reviewer.review)
    graph.add_node("finalizer", finalizer.finalize)
    graph.add_node("consistency_checker", consistency_checker.check)
    graph.add_node("reference_generator", reference_generator.generate)

    graph.add_edge(START, "topic_proposer")
    graph.add_edge("topic_proposer", "researcher")
    graph.add_edge("researcher", "planner")
    graph.add_edge("planner", "writer")
    graph.add_edge("writer", "reviewer")

    def route_reviewer(state: AgentState) -> Literal["writer", "finalizer"]:
        sections = state["sections"]
        index = state["current_section_index"]

        if sections[index].status == "approved":
            if index + 1 >= len(sections):
                return "finalizer"
            return "writer"
        return "writer"

    graph.add_conditional_edges(
        "reviewer", route_reviewer, {"writer": "writer", "finalizer": "finalizer"}
    )
    graph.add_edge("finalizer", "reference_generator")
    graph.add_edge("reference_generator", "consistency_checker")
    graph.add_edge("consistency_checker", END)

    return graph.compile(checkpointer=MemorySaver(), interrupt_after=["topic_proposer", "planner"])


def create_simple_writing_graph(
    model: ChatOpenAI,
    prompts: PromptsConfig,
    rag_retriever: RAGRetrieverFunc | None = None,
    semantic_scholar_key: str | None = None,
    openalex_email: str | None = None,
    extra_paper_retrievers: list[PaperSearchFunc] | None = None,
) -> CompiledStateGraph:
    topic_proposer = TopicProposerNode(llm=model, prompts=prompts)
    researcher = ResearcherNode(
        llm=model,
        prompts=prompts,
        semantic_scholar_key=semantic_scholar_key or settings.SEMANTIC_SCHOLAR_API_KEY,
        openalex_email=openalex_email,
        extra_paper_retrievers=extra_paper_retrievers,
    )
    planner = PlannerNode(llm=model, prompts=prompts)
    writer = WriterNode(llm=model, prompts=prompts, rag_retriever=rag_retriever)
    reviewer = ReviewerNode(llm=model, prompts=prompts)
    finalizer = FinalizerNode(llm=model, prompts=prompts)
    consistency_checker = ConsistencyCheckerNode(llm=model, prompts=prompts)
    reference_generator = ReferenceGeneratorNode()

    graph = StateGraph(AgentState)

    graph.add_node("proposer", topic_proposer.propose)
    graph.add_node("researcher", researcher.search)
    graph.add_node("planner", planner.plan)
    graph.add_node("writer", writer.write)
    graph.add_node("reviewer", reviewer.review)
    graph.add_node("finalizer", finalizer.finalize)
    graph.add_node("consistency_checker", consistency_checker.check)
    graph.add_node("reference_generator", reference_generator.generate)

    graph.add_edge(START, "proposer")
    graph.add_edge("proposer", "researcher")
    graph.add_edge("researcher", "planner")
    graph.add_edge("planner", "writer")
    graph.add_edge("writer", "reviewer")

    def should_continue(state: AgentState) -> Literal["writer", "finalizer"]:
        sections = state["sections"]
        idx = state["current_section_index"]
        if sections[idx].status == "approved" and idx + 1 >= len(sections):
            return "finalizer"
        return "writer"

    graph.add_conditional_edges(
        "reviewer", should_continue, {"writer": "writer", "finalizer": "finalizer"}
    )
    graph.add_edge("finalizer", "reference_generator")
    graph.add_edge("reference_generator", "consistency_checker")
    graph.add_edge("consistency_checker", END)

    return graph.compile(checkpointer=MemorySaver())
