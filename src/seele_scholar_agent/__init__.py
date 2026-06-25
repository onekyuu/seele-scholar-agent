from .agent_config import PaperSearchFunc, RAGRetrieverFunc
from .evals import QualityMetrics, evaluate_quality
from .graph import create_simple_writing_graph, create_writing_graph
from .logging import setup_logging
from .nodes import NodeStreamEvent
from .nodes.consistency_checker import ConsistencyCheckerNode
from .nodes.finalizer import FinalizerNode
from .nodes.integrity_gate import IntegrityGateNode
from .nodes.outline_quality_gate import OutlineQualityGateNode
from .nodes.planner import PlannerNode
from .nodes.reference_generator import ReferenceGeneratorNode
from .nodes.researcher import ResearcherNode
from .nodes.reviewer import ReviewerNode
from .nodes.topic_proposer import TopicProposerNode
from .nodes.writer import WriterNode
from .state import (
    AgentState,
    ClaimEvidenceBinding,
    EvidencePacket,
    MaterialRegistry,
    MaterialRegistryEntry,
    OutlineStructure,
    PaperMetadata,
    QualityIssue,
    SectionDraft,
    SectionStyleGuidance,
)
from .style_packs import LocaleStylePack, StyleReference

__version__ = "0.12.5"

__all__ = [
    "create_writing_graph",
    "create_simple_writing_graph",
    "setup_logging",
    "NodeStreamEvent",
    "TopicProposerNode",
    "ResearcherNode",
    "PlannerNode",
    "WriterNode",
    "ReviewerNode",
    "FinalizerNode",
    "ConsistencyCheckerNode",
    "ReferenceGeneratorNode",
    "IntegrityGateNode",
    "OutlineQualityGateNode",
    "AgentState",
    "PaperMetadata",
    "QualityIssue",
    "EvidencePacket",
    "ClaimEvidenceBinding",
    "MaterialRegistry",
    "MaterialRegistryEntry",
    "OutlineStructure",
    "SectionDraft",
    "SectionStyleGuidance",
    "LocaleStylePack",
    "StyleReference",
    "RAGRetrieverFunc",
    "PaperSearchFunc",
    "QualityMetrics",
    "evaluate_quality",
]
