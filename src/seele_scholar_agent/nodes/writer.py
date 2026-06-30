import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig, RAGRetrieverFunc
from ..config import settings
from ..document_profile import (
    is_compound_section_title,
    is_research_proposal,
    is_schedule_section,
    missing_proposal_core_tasks,
)
from ..i18n import t
from ..logging import get_logger
from ..policy import SectionExecutionStrategy
from ..state import (
    AgentState,
    ClaimEvidenceBinding,
    DocumentChunk,
    EvidencePacket,
    PaperMetadata,
    SectionDraft,
    SectionOutline,
)
from ..style_packs import build_writer_style_context
from ..writing import SectionWritingSpec, WriterInput, WriterInputBuilder
from . import (
    CITATION_PATTERN,
    PREVIOUS_SECTION_MAX_CHARS,
    SECTION_SUMMARY_MAX_CHARS,
    NodeStreamEvent,
    _stream_llm_text,
    invoke_with_retry,
)
from .claim_audit import RuleBasedClaimExtractor
from .material_registry import (
    annotate_paper_summaries,
    get_material_registry,
    material_policy_suffix,
)

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")
_BUDGET_RE = re.compile(
    r"(\d+\s*(?:-\s*\d+\s*)?(?:字|文字|語|words?)|全文|预算|予算|budget|target_words|字数|字數)",
    re.IGNORECASE,
)

_PROPOSAL_REVISION_USER_PROMPT = """研究主题：{topic}

目标语言：{language}

当前章节：{section_title}
章节描述与长度约束：
{section_description}

论文/计划书大纲：
{outline_json}

已完成章节摘要（用于保持前后连贯，避免重复）：
{previous_sections}

当前版本正文（必须在此基础上返修，不要自由重写成无关版本）：
{current_content}

审稿意见（优先处理 blocking；warning 可用压缩、弱化断言或删去不必要引用解决）：
{review_comments}

可引用论文列表（引用时只能使用以下编号，格式为 [N]）：
{numbered_papers}

相关文献证据包：
{rag_context}

语言与文体指导：
{style_guidance}

请为 research_proposal（日本大学院研究計画書）输出该章节的完整替换正文，不要输出章节标题。

返修要求：
1. 优先修复 blocking issue：缺少标题核心任务、结构断裂、数字枚举不一致、内容截断、
   目的/方法/计划无法判断可行性。不要机械扩写所有 reviewer comments。
2. 保留章节在预算内的完整结构；若篇幅紧张，压缩背景和铺垫，优先保留研究主题、
   目的、方法可行性和计划。
3. 申请者自己的研究计划、时间安排、拟开展实验和执行安排不需要强行加引用；
   引用只用于文献事实或已有研究结论。
4. 对非阻塞 missing_citation、citation_mismatch、unsupported claim，可通过弱化断言、
   删除不必要引用、改为更谨慎表达解决，不要为了补引用而越写越长。
5. 「研究方法」只需概要说明使用什么资料/工具/方法、如何验证或分析、为什么硕士阶段
   可行；不要求完整实验 protocol、详细变量设计、统计检验细节。
6. 「期待される成果」只需说明 1-2 个预期贡献或申请价值，不要写成论文 contribution
   section。
7. 避免模板式声明“三点”“三段階”，除非正文实际写出对应数量。
8. 若本章节是明确的「研究計画・スケジュール」或 schedule/timeline 章节，必须覆盖
   1年次前期、1年次後期、2年次前期、2年次後期。每个阶段至少写出具体任务
   和交付物/里程碑；即使预算很紧，也不允许省略 2 年次。
9. 直接输出正文，不要解释修改过程，不要使用 # 或 ## 标题符号。"""

_PROPOSAL_DRAFT_USER_PROMPT = """研究主题：{topic}

目标语言：{language}

当前章节：{section_title}
章节描述与长度约束：
{section_description}

研究计划书大纲：
{outline_json}

已完成章节摘要（供上下文参考，避免重复）：
{previous_sections}

可引用论文列表（引用时只能使用以下编号，格式为 [N]）：
{numbered_papers}

相关文献证据包：
{rag_context}

语言与文体指导：
{style_guidance}

请为 research_proposal（日本大学院研究計画書）撰写该章节正文，不要输出章节标题。

写作要求：
1. 按申请材料而非学术论文正文写作，突出研究动机、研究目的、研究方法、
   可行性、计划性和申请者自身的问题意识。面向教授快速判断选题价值与可行性。
2. 申请者自己的计划、拟开展实验、时间安排、交付物和将来展望不需要引用。
   引用只用于先行研究、背景事实或已有研究结论。
3. 遵守章节描述和 outline 中的 target_words/总字数预算；篇幅紧张时压缩背景，
   优先保留研究主题、目的、方法可行性和计划，不要省略章节核心任务。
4. 「研究方法」只需概要级说明：使用什么资料/工具/方法，如何验证或分析，为什么
   硕士阶段可行。不要求完整实验 protocol、详细变量设计或统计检验细节。
5. 「期待される成果」只需说明 1-2 个预期贡献或申请价值，不要求完整论文式
   contribution section。
6. 复合标题只需让两侧都有概要级信息，不要把每一侧扩写成完整论文小节。
7. 避免模板式声明“三点”“三段階”，除非实际写出对应数量。
8. 若本章节是明确的「研究計画・スケジュール」或 schedule/timeline 章节，必须覆盖
   1年次前期、1年次後期、2年次前期、2年次後期。每个阶段至少写出具体任务
   和交付物/里程碑。
9. 直接输出正文，不要解释写作过程，不要使用 # 或 ## 标题符号。"""

_ACADEMIC_REVISION_USER_PROMPT = """论文主题：{topic}

目标语言：{language}

当前章节：{section_title}
章节描述与长度约束：
{section_description}

论文大纲：
{outline_json}

已完成章节摘要（用于保持前后连贯，避免重复）：
{previous_sections}

当前版本正文（必须在此基础上返修，不要自由重写成无关版本）：
{current_content}

审稿意见（逐条消化，每条都必须在新正文中得到处理）：
{review_comments}

可引用论文列表（引用时只能使用以下编号，格式为 [N]）：
{numbered_papers}

相关文献证据包（事实性主张必须尽量绑定具体证据；不要编造来源）：
{rag_context}

语言与文体指导：
{style_guidance}

请输出该学术论文章节的完整替换正文，不要输出章节标题。

返修要求：
1. 逐条解决审稿意见；优先替换、压缩、合并、精确化现有段落，
   不要只在末尾追加补丁。
2. 保留并改进当前正文中已正确、已被引用支撑的内容。
3. 对 missing_citation、citation_mismatch、unsupported claim，必须删除、
   降级或改写无证据主张；需要保留的事实性主张必须使用有效 [N] 引用。
4. 对 methodology/statistics 意见，补足样本量、数据集、baseline、指标定义、
   显著性/不确定性或实验边界；若无法由证据支持，应改写为保守表述。
5. 保持章节结构完整、段落衔接清楚，并遵守章节描述中的长度预算。
6. 直接输出正文，不要解释修改过程，不要使用 # 或 ## 标题符号。"""


@dataclass(frozen=True)
class PacketSelection:
    packet: EvidencePacket | None
    diagnostics: dict[str, Any]


def _generate_section_summary(title: str, content: str) -> str:
    """Generate a compact summary (~150 tokens) of a written section for use as prior context.

    Uses a heuristic paragraph extraction — no extra LLM call needed.
    """
    if not content:
        return f"[{title}]\n(empty)"

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        snippet = content[:SECTION_SUMMARY_MAX_CHARS]
        if len(content) > SECTION_SUMMARY_MAX_CHARS:
            snippet += "..."
        return f"[{title}]\n{snippet}"

    parts: list[str] = []
    total_chars = 0
    for para in paragraphs:
        remaining = SECTION_SUMMARY_MAX_CHARS - total_chars
        if remaining <= 0:
            break
        if len(para) <= remaining:
            parts.append(para)
            total_chars += len(para)
        else:
            # End at a sentence boundary when possible
            snippet = para[:remaining]
            last_period = snippet.rfind(". ")
            if last_period > remaining // 2:
                snippet = snippet[: last_period + 1]
            else:
                snippet += "..."
            parts.append(snippet)
            break

    return f"[{title}]\n" + "\n\n".join(parts)


def _metadata_str(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _metadata_str_list(metadata: dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        if isinstance(value, str) and value.strip():
            return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _metadata_int(metadata: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _metadata_float(metadata: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in _WORD_RE.findall(left)}
    right_tokens = {token.lower() for token in _WORD_RE.findall(right)}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


class DraftWriter:
    def __init__(self, chain: Any):
        self.chain = chain

    async def draft(self, input_data: dict[str, Any]) -> str:
        result = await invoke_with_retry(self.chain, input_data)
        content = result.content if hasattr(result, "content") else str(result)
        if isinstance(content, list):
            return "\n".join(str(c) for c in content)
        return str(content)


class StylePolisher:
    def polish(self, content: str) -> str:
        lines = content.split("\n")
        clean = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                clean.append(line)
                continue
            if line.strip():
                clean.append(line)

        return "\n".join(clean).strip()


class CitationBinder:
    def __init__(self) -> None:
        self.claim_extractor = RuleBasedClaimExtractor()

    def bind(
        self,
        section: SectionDraft,
        content: str,
        papers: list[PaperMetadata],
        evidence_packets: list[EvidencePacket],
    ) -> list[ClaimEvidenceBinding]:
        bindings: list[ClaimEvidenceBinding] = []
        seen: set[tuple[str, int]] = set()
        for claim_text in self._extract_cited_claims(content):
            for citation_number in {int(n) for n in CITATION_PATTERN.findall(claim_text)}:
                key = (claim_text, citation_number)
                if key in seen:
                    continue
                seen.add(key)
                selection = self._select_packet(
                    citation_number, claim_text, papers, evidence_packets
                )
                packet = selection.packet
                score = self._support_score(claim_text, packet)
                bindings.append(
                    ClaimEvidenceBinding(
                        section_id=section.section_id,
                        claim_text=claim_text,
                        citation_number=citation_number,
                        chunk_id=packet.chunk_id if packet else None,
                        source_paper_id=self._source_paper_id(citation_number, papers, packet),
                        support_score=score,
                        verdict=self._verdict(score, packet),
                        diagnostics=selection.diagnostics,
                    )
                )
        return bindings

    def _extract_cited_claims(self, content: str) -> list[str]:
        matches = [claim.text for claim in self.claim_extractor.extract_cited_claims(content)]
        if matches:
            return matches
        return [line.strip() for line in content.splitlines() if CITATION_PATTERN.search(line)]

    def _select_packet(
        self,
        citation_number: int,
        claim_text: str,
        papers: list[PaperMetadata],
        evidence_packets: list[EvidencePacket],
    ) -> PacketSelection:
        source_paper_id = None
        if 1 <= citation_number <= len(papers):
            source_paper_id = papers[citation_number - 1].paper_id

        diagnostics: dict[str, Any] = {
            "evidence_packet_count": len(evidence_packets),
            "candidate_count": 0,
            "source_paper_id_match": False,
            "source_paper_id": source_paper_id,
        }
        if not evidence_packets:
            return PacketSelection(packet=None, diagnostics=diagnostics)

        candidates = [
            packet for packet in evidence_packets if packet.source_paper_id == source_paper_id
        ]
        diagnostics["source_paper_id_match"] = source_paper_id is not None and bool(candidates)
        if not candidates:
            candidates = evidence_packets
        diagnostics["candidate_count"] = len(candidates)

        packet = max(
            candidates,
            key=lambda packet: (
                _token_overlap_score(claim_text, packet.quote),
                packet.relevance_score,
            ),
        )
        return PacketSelection(packet=packet, diagnostics=diagnostics)

    def _source_paper_id(
        self, citation_number: int, papers: list[PaperMetadata], packet: EvidencePacket | None
    ) -> str | None:
        if packet and packet.source_paper_id:
            return packet.source_paper_id
        if 1 <= citation_number <= len(papers):
            return papers[citation_number - 1].paper_id
        return None

    def _support_score(self, claim_text: str, packet: EvidencePacket | None) -> float:
        if packet is None:
            return 0.0
        return max(_token_overlap_score(claim_text, packet.quote), packet.relevance_score * 0.5)

    def _verdict(
        self, score: float, packet: EvidencePacket | None
    ) -> Literal["supported", "weak", "unsupported", "unverified"]:
        if packet is None:
            return "unverified"
        if score >= settings.CITATION_BINDER_SUPPORTED_THRESHOLD:
            return "supported"
        if score >= settings.CITATION_BINDER_WEAK_THRESHOLD:
            return "weak"
        return "unsupported"


class WriterNode:
    def __init__(
        self,
        llm: ChatOpenAI,
        prompts: PromptsConfig,
        rag_retriever: RAGRetrieverFunc | None = None,
        execution_strategy: SectionExecutionStrategy | None = None,
    ):
        self.llm = llm
        self.prompts = prompts
        self.rag_retriever = rag_retriever
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", prompts.writer_system_prompt), ("user", prompts.writer_user_prompt)]
        )
        self.proposal_revision_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.writer_system_prompt),
                ("user", prompts.proposal_revision_user_prompt or _PROPOSAL_REVISION_USER_PROMPT),
            ]
        )
        self.proposal_draft_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.writer_system_prompt),
                ("user", prompts.proposal_writer_user_prompt or _PROPOSAL_DRAFT_USER_PROMPT),
            ]
        )
        self.academic_revision_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.writer_system_prompt),
                ("user", prompts.academic_revision_user_prompt or _ACADEMIC_REVISION_USER_PROMPT),
            ]
        )
        self.chain = self.prompt | self.llm
        self.proposal_draft_chain = self.proposal_draft_prompt | self.llm
        self.proposal_revision_chain = self.proposal_revision_prompt | self.llm
        self.academic_revision_chain = self.academic_revision_prompt | self.llm
        self.draft_writer = DraftWriter(self.chain)
        self.proposal_draft_writer = DraftWriter(self.proposal_draft_chain)
        self.proposal_revision_writer = DraftWriter(self.proposal_revision_chain)
        self.academic_revision_writer = DraftWriter(self.academic_revision_chain)
        self.citation_binder = CitationBinder()
        self.style_polisher = StylePolisher()
        self.execution_strategy = execution_strategy or SectionExecutionStrategy()
        self.writer_input_builder = WriterInputBuilder()

    async def write(self, state: AgentState) -> dict[str, Any]:
        sections = state["sections"]
        current_index = state["current_section_index"]
        lang = state.get("language", "zh")

        if current_index >= len(sections):
            return {
                "status": "completed",
                "sections_completed": [
                    s.title
                    for s in sections
                    if s.status in {"approved", "accepted_with_issues"}
                ],
            }

        section = sections[current_index]
        proposal_profile = is_research_proposal(state)
        writer_mode = self._writer_mode(state, section)
        revision_mode = self._has_revision_context(section)

        if section.status == "approved":
            return await self._move_to_next(state)

        logger.info("writing section", title=section.title, language=lang)

        evidence_packets, new_evidence_packets = await self._collect_evidence_packets(
            state, section
        )
        rag_context = self._build_rag_context(evidence_packets)

        section_outline = self._find_section_outline(state, section)
        style_guidance = build_writer_style_context(state, section_outline)
        writer_input = self.writer_input_builder.build(
            state,
            current_index=current_index,
            evidence_packets=evidence_packets,
            style_context=style_guidance,
        )

        paper_summaries: list[str] = state.get("paper_summaries") or []
        numbered_papers = self._build_citable_sources(
            state,
            paper_summaries=paper_summaries,
        )

        try:
            input_data = self._build_prompt_input(
                writer_input,
                section,
                lang,
                numbered_papers,
                rag_context,
            )
            writer = self._draft_writer_for_mode(writer_mode)
            raw_content = await writer.draft(input_data)
            content = self.style_polisher.polish(raw_content)
        except Exception as e:
            logger.error("writing failed after retries", error=str(e))
            updated_sections = sections.copy()
            updated_sections[current_index] = section.model_copy(update={"status": "pending"})
            return {
                "sections": updated_sections,
                "status": "failed",
                "error_message": f"Writing section '{section.title}' failed: {e}",
            }

        updated_sections = sections.copy()
        updated_sections[current_index] = section.model_copy(
            update={
                "content": content,
                "status": "review",
                "revision_count": section.revision_count,
            }
        )
        claim_evidence_bindings = self.citation_binder.bind(
            section, content, state.get("papers", []), evidence_packets
        )
        effective_bindings = [
            binding
            for binding in state.get("claim_evidence_bindings", [])
            if binding.section_id != section.section_id
        ]
        effective_bindings.extend(claim_evidence_bindings)

        updated_summaries = list(state.get("section_summaries") or [])
        while len(updated_summaries) <= current_index:
            updated_summaries.append("")
        updated_summaries[current_index] = _generate_section_summary(section.title, content)

        result: dict[str, Any] = {
            "sections": updated_sections,
            "section_summaries": updated_summaries,
            "claim_evidence_bindings": effective_bindings,
            "writer_input": writer_input,
            "status": "reviewing",
            "writer_diagnostics": {
                "revision_mode": revision_mode,
                "writer_mode": writer_mode,
                "proposal_profile": proposal_profile,
                "section_title": section.title,
                "section_budget": self._budget_diagnostic_value(
                    writer_input.current_section
                ),
                "section_budget_spec": self._budget_spec_diagnostic_value(
                    writer_input.current_section
                ),
                "compound_title_detected": is_compound_section_title(section.title),
                "missing_core_tasks": (
                    missing_proposal_core_tasks(section.title, content)
                    if proposal_profile
                    else []
                ),
                "review_comment_count": len(section.review_comments),
                "schedule_section": is_schedule_section(section.title),
            },
        }
        if new_evidence_packets:
            result["evidence_packets"] = new_evidence_packets
        return result

    async def astream(self, state: AgentState) -> AsyncIterator[NodeStreamEvent]:
        sections = state["sections"]
        current_index = state["current_section_index"]
        lang = state.get("language", "zh")

        if current_index >= len(sections):
            yield NodeStreamEvent(
                type="result",
                result={
                    "status": "completed",
                    "sections_completed": [
                        s.title
                        for s in sections
                        if s.status in {"approved", "accepted_with_issues"}
                    ],
                },
            )
            return

        section = sections[current_index]
        proposal_profile = is_research_proposal(state)
        writer_mode = self._writer_mode(state, section)
        revision_mode = self._has_revision_context(section)

        if section.status == "approved":
            move_result = await self._move_to_next(state)
            yield NodeStreamEvent(type="result", result=move_result)
            return

        yield NodeStreamEvent(type="progress", progress=f"writing:{section.title}")

        evidence_packets, new_evidence_packets = await self._collect_evidence_packets(
            state, section
        )
        rag_context = self._build_rag_context(evidence_packets)

        _section_summaries: list[str] = list(state.get("section_summaries") or [])
        _paper_summaries: list[str] = state.get("paper_summaries") or []

        style_guidance = build_writer_style_context(
            state, self._find_section_outline(state, section)
        )
        writer_input = self.writer_input_builder.build(
            state,
            current_index=current_index,
            evidence_packets=evidence_packets,
            style_context=style_guidance,
        )

        input_data = self._build_prompt_input(
            writer_input,
            section,
            lang,
            (
                self._build_citable_sources(
                    state,
                    paper_summaries=_paper_summaries,
                )
            ),
            rag_context,
        )

        full_text = ""
        stream_chain = self._stream_chain_for_mode(writer_mode)
        async for event in _stream_llm_text(stream_chain, input_data):
            full_text += event.get("token", "")
            yield event

        content = self._clean_content(full_text)
        claim_evidence_bindings = self.citation_binder.bind(
            section, content, state.get("papers", []), evidence_packets
        )
        effective_bindings = [
            binding
            for binding in state.get("claim_evidence_bindings", [])
            if binding.section_id != section.section_id
        ]
        effective_bindings.extend(claim_evidence_bindings)

        updated_sections = sections.copy()
        updated_sections[current_index] = section.model_copy(
            update={
                "content": content,
                "status": "review",
                "revision_count": section.revision_count,
            }
        )

        _updated_summaries = list(state.get("section_summaries") or [])
        while len(_updated_summaries) <= current_index:
            _updated_summaries.append("")
        _updated_summaries[current_index] = _generate_section_summary(section.title, content)

        result: dict[str, Any] = {
            "sections": updated_sections,
            "section_summaries": _updated_summaries,
            "claim_evidence_bindings": effective_bindings,
            "writer_input": writer_input,
            "status": "reviewing",
            "writer_diagnostics": {
                "revision_mode": revision_mode,
                "writer_mode": writer_mode,
                "proposal_profile": proposal_profile,
                "section_title": section.title,
                "section_budget": self._budget_diagnostic_value(
                    writer_input.current_section
                ),
                "section_budget_spec": self._budget_spec_diagnostic_value(
                    writer_input.current_section
                ),
                "compound_title_detected": is_compound_section_title(section.title),
                "missing_core_tasks": (
                    missing_proposal_core_tasks(section.title, content)
                    if proposal_profile
                    else []
                ),
                "review_comment_count": len(section.review_comments),
                "schedule_section": is_schedule_section(section.title),
            },
        }
        if new_evidence_packets:
            result["evidence_packets"] = new_evidence_packets

        yield NodeStreamEvent(type="result", result=result)

    def _build_prompt_input(
        self,
        writer_input: WriterInput,
        section: SectionDraft,
        lang: str,
        numbered_papers: str,
        rag_context: str,
    ) -> dict[str, Any]:
        return {
            "topic": writer_input.topic,
            "language": t(lang, "language_name"),
            "section_title": writer_input.current_section.title,
            "section_description": self._build_section_description_from_spec(
                writer_input.current_section
            ),
            "suggested_figures": self._build_suggested_figures_from_spec(
                writer_input.current_section
            ),
            "outline_json": self._build_outline_from_writer_input(writer_input),
            "previous_sections": self._previous_summaries_text(writer_input),
            "numbered_papers": numbered_papers,
            "rag_context": rag_context,
            "style_guidance": self._style_guidance_with_exemplar_context(writer_input),
            "review_comments": self._review_comments_text(writer_input),
            "current_content": section.content or "无",
        }

    def _build_section_description_from_spec(self, spec: SectionWritingSpec) -> str:
        description = spec.description.strip()
        constraints: list[str] = []
        if spec.budget is not None:
            budget_parts: list[str] = []
            if spec.budget.target is not None:
                budget_parts.append(f"target={spec.budget.target}")
            if spec.budget.hard_limit is not None:
                budget_parts.append(f"hard_limit={spec.budget.hard_limit}")
            budget_parts.append(f"unit={spec.budget.unit}")
            constraints.append("Length budget: " + ", ".join(budget_parts) + ".")
        elif _BUDGET_RE.search(description):
            constraints.append(
                "Length constraint: follow the explicit budget in the section description "
                "as the only hard length limit; ignore generic 200-500 defaults."
            )
        else:
            constraints.append(
                "Default length target: 200-500 words/characters as appropriate for the "
                "target language, unless the outline or caller provides a stricter budget."
            )

        if spec.purpose:
            constraints.append(f"Purpose: {spec.purpose}")
        if spec.content_summary:
            constraints.append(f"Content summary: {spec.content_summary}")
        if spec.target_claims:
            constraints.append("Target claims: " + "; ".join(spec.target_claims))
        if spec.citation_plan:
            constraints.append("Citation plan: " + "; ".join(spec.citation_plan))
        if is_schedule_section(spec.title):
            constraints.append(
                "Schedule density: use compact wording but preserve the complete two-year "
                "timeline, including 1年次前期, 1年次後期, 2年次前期, and 2年次後期."
            )
        if not description:
            return "\n".join(constraints)
        return f"{description}\n\n" + "\n".join(constraints)

    def _build_suggested_figures_from_spec(self, spec: SectionWritingSpec) -> str:
        if not spec.suggested_figures:
            return "无"
        return "\n".join(f"- {figure}" for figure in spec.suggested_figures)

    def _build_outline_from_writer_input(self, writer_input: WriterInput) -> str:
        outline = writer_input.outline_context
        lines = [
            f"Title: {outline.title}",
            f"Paper type: {outline.paper_type}",
            f"Structure pattern: {outline.structure_pattern}",
            "",
        ]
        for section in outline.sections:
            lines.append(f"- {section.title}")
            if section.purpose:
                lines.append(f"  Purpose: {section.purpose}")
            if section.content_summary:
                lines.append(f"  Content summary: {section.content_summary}")

        current = writer_input.current_section
        lines.extend(["", "Current section writing specification:"])
        lines.append(f"- {current.title}: {current.description}")
        if current.key_points:
            lines.append(f"  Key points: {'; '.join(current.key_points)}")
        if current.target_claims:
            lines.append(f"  Target claims: {'; '.join(current.target_claims)}")
        if current.key_sources:
            lines.append(f"  Key sources: {'; '.join(current.key_sources)}")
        if current.evidence_gaps:
            lines.append(f"  Evidence gaps: {'; '.join(current.evidence_gaps)}")
        if current.transition_to_next:
            lines.append(f"  Transition: {current.transition_to_next}")
        return "\n".join(lines)

    def _previous_summaries_text(self, writer_input: WriterInput) -> str:
        if not writer_input.previous_section_summaries:
            return "无"
        return "\n\n---\n\n".join(writer_input.previous_section_summaries)

    def _review_comments_text(self, writer_input: WriterInput) -> str:
        if not writer_input.review_comments:
            return "无"
        return "\n".join(f"- {comment}" for comment in writer_input.review_comments)

    def _style_guidance_with_exemplar_context(self, writer_input: WriterInput) -> str:
        parts = [writer_input.style_context.strip()] if writer_input.style_context.strip() else []
        exemplar_context = writer_input.exemplar_context
        if exemplar_context is None:
            return "\n\n".join(parts) if parts else "无"

        exemplar_lines: list[str] = []
        if exemplar_context.outline_patterns:
            exemplar_lines.append("Outline patterns:")
            exemplar_lines.extend(f"- {pattern}" for pattern in exemplar_context.outline_patterns)
        if exemplar_context.style_notes:
            exemplar_lines.append("Style notes:")
            exemplar_lines.extend(f"- {note}" for note in exemplar_context.style_notes)
        if exemplar_context.section_examples:
            exemplar_lines.append("Section examples:")
            for example in exemplar_context.section_examples:
                snippet = example.text[:500]
                if len(example.text) > 500:
                    snippet += "..."
                label = example.section_title or example.section_role or example.chunk_id
                exemplar_lines.append(f"- [{example.chunk_id}] {label}: {snippet}")
        anti_copying_notes = exemplar_context.anti_copying_notes or [
            "Use exemplar materials only as structure/style references; do not copy wording."
        ]
        exemplar_lines.append("Anti-copying notes:")
        exemplar_lines.extend(f"- {note}" for note in anti_copying_notes)

        if exemplar_lines:
            parts.append(
                "Exemplar context (reference only, do not reuse wording):\n"
                + "\n".join(exemplar_lines)
            )
        return "\n\n".join(parts) if parts else "无"

    def _budget_diagnostic_value(self, spec: SectionWritingSpec) -> int | None:
        if spec.budget is None:
            return None
        if spec.budget.hard_limit is None:
            return spec.budget.target
        return spec.budget.hard_limit

    def _budget_spec_diagnostic_value(self, spec: SectionWritingSpec) -> dict[str, Any] | None:
        if spec.budget is None:
            return None
        return spec.budget.model_dump()

    def _build_draft_input(
        self,
        state: AgentState,
        section: SectionDraft,
        lang: str,
        outline_json: str,
        previous_sections: str,
        numbered_papers: str,
        rag_context: str,
        style_guidance: str,
        review_comments: str,
    ) -> dict[str, Any]:
        return {
            "topic": state["topic"],
            "language": t(lang, "language_name"),
            "section_title": section.title,
            "section_description": self._build_section_description(section),
            "suggested_figures": self._build_suggested_figures(section, state),
            "outline_json": outline_json,
            "previous_sections": previous_sections,
            "numbered_papers": numbered_papers,
            "rag_context": rag_context,
            "style_guidance": style_guidance,
            "review_comments": review_comments,
            "current_content": section.content or "无",
        }

    def _find_section_outline(
        self, state: AgentState, section: SectionDraft
    ) -> SectionOutline | None:
        outline = state.get("outline")
        if outline is None:
            return None
        return next(
            (
                section_outline
                for section_outline in outline.sections
                if section_outline.title == section.title
            ),
            None,
        )

    async def _collect_evidence_packets(
        self, state: AgentState, section: SectionDraft
    ) -> tuple[list[EvidencePacket], list[EvidencePacket]]:
        if self.rag_retriever:
            search_query = f"{state['topic']} {section.title} {section.description}"
            rag_chunks = await self.rag_retriever(search_query)
            packets = self._chunks_to_evidence_packets(
                rag_chunks, section_title=section.title, why_relevant=search_query
            )
            return packets, packets

        existing_packets = list(state.get("evidence_packets") or [])
        if existing_packets:
            return existing_packets, []

        packets = self._chunks_to_evidence_packets(
            state.get("rag_context", []),
            section_title=section.title,
            why_relevant=f"{state['topic']} {section.title}",
        )
        return packets, packets

    def _chunks_to_evidence_packets(
        self, chunks: list[DocumentChunk], section_title: str, why_relevant: str
    ) -> list[EvidencePacket]:
        packets: list[EvidencePacket] = []
        for chunk in chunks:
            metadata = chunk.metadata
            quote = _metadata_str(metadata, "quote", "text", "content") or chunk.content
            packets.append(
                EvidencePacket(
                    chunk_id=chunk.chunk_id,
                    title=_metadata_str(metadata, "title", "paper_title"),
                    authors=_metadata_str_list(metadata, "authors", "author"),
                    year=_metadata_int(metadata, "year", "publication_year"),
                    page=_metadata_str(metadata, "page", "pages") or None,
                    section=_metadata_str(metadata, "section") or section_title,
                    source=chunk.source,
                    source_paper_id=_metadata_str(
                        metadata, "paper_id", "source_paper_id", "corpus_id"
                    )
                    or None,
                    relevance_score=_metadata_float(metadata, "relevance_score", "score"),
                    why_relevant=_metadata_str(metadata, "why_relevant") or why_relevant,
                    quote=quote,
                )
            )
        return packets

    def _build_suggested_figures(self, section: SectionDraft, state: AgentState) -> str:
        outline = state.get("outline")
        if not outline:
            return "无"
        for sec_outline in outline.sections:
            if sec_outline.title == section.title and sec_outline.suggested_figures:
                lines = [f"- {fig}" for fig in sec_outline.suggested_figures]
                return "\n".join(lines)
        return "无"

    def _build_outline_json(self, outline: Any) -> str:
        if not outline:
            return ""
        lines = [
            f"Title: {outline.title}",
            f"Paper type: {getattr(outline, 'paper_type', 'auto')}",
            f"Structure pattern: {getattr(outline, 'structure_pattern', 'auto')}",
            "",
        ]
        for s in outline.sections:
            lines.append(f"- {s.title}: {s.description}")
            purpose = getattr(s, "purpose", "")
            if purpose:
                lines.append(f"  Purpose: {purpose}")
            content_summary = getattr(s, "content_summary", "")
            if content_summary:
                lines.append(f"  Content summary: {content_summary}")
            target_words = getattr(s, "target_words", None)
            if target_words:
                lines.append(f"  Target words/chars: {target_words}")
            target_claims = getattr(s, "target_claims", [])
            if target_claims:
                lines.append(f"  Target claims: {'; '.join(target_claims)}")
            key_sources = getattr(s, "key_sources", [])
            if key_sources:
                lines.append(f"  Key sources: {'; '.join(key_sources)}")
            evidence_gaps = getattr(s, "evidence_gaps", [])
            if evidence_gaps:
                lines.append(f"  Evidence gaps: {'; '.join(evidence_gaps)}")
            transition = getattr(s, "transition_to_next", "")
            if transition:
                lines.append(f"  Transition: {transition}")
            section_style = getattr(s, "section_style", None)
            if section_style is not None:
                if section_style.argument_mode:
                    lines.append(f"  Argument mode: {section_style.argument_mode}")
                if section_style.sentence_style:
                    lines.append(f"  Sentence style: {section_style.sentence_style}")
                if section_style.transition_style:
                    lines.append(f"  Transition style: {section_style.transition_style}")
                if section_style.forbidden_patterns:
                    lines.append(
                        "  Forbidden patterns: "
                        + "; ".join(section_style.forbidden_patterns)
                    )
        return "\n".join(lines)

    def _build_rag_context(self, rag_context: Any) -> str:
        if not rag_context:
            return "无"
        parts = []
        for c in rag_context[:5]:
            if isinstance(c, EvidencePacket):
                authors = ", ".join(c.authors) if c.authors else "Unknown"
                parts.append(
                    "\n".join(
                        [
                            f"[chunk_id:{c.chunk_id}]",
                            f"title: {c.title or 'Unknown'}",
                            f"authors: {authors}",
                            f"year: {c.year or 'Unknown'}",
                            f"page: {c.page or 'N/A'}",
                            f"section: {c.section or 'N/A'}",
                            f"relevance_score: {c.relevance_score:.2f}",
                            f"why_relevant: {c.why_relevant or 'N/A'}",
                            f"quote: {c.quote}",
                        ]
                    )
                )
            else:
                parts.append(f"[chunk_id:{c.chunk_id}]\n{c.content}")
        return "\n\n".join(parts)

    def _build_review_comments(self, section: Any) -> str:
        if not section.review_comments:
            return "无"
        return "\n".join([f"- {c}" for c in section.review_comments])

    def _build_section_description(self, section: SectionDraft) -> str:
        description = section.description.strip()
        constraints: list[str] = []

        if _BUDGET_RE.search(description):
            constraints.append(
                "Length constraint: follow the explicit budget in the section description "
                "as the only hard length limit; ignore generic 200-500 defaults."
            )
        else:
            constraints.append(
                "Default length target: 200-500 words/characters as appropriate for the "
                "target language, unless the outline or caller provides a stricter budget."
            )

        if is_schedule_section(section.title):
            constraints.append(
                "Schedule density: use compact wording but preserve the complete two-year "
                "timeline, including 1年次前期, 1年次後期, 2年次前期, and 2年次後期."
            )

        if not description:
            return "\n".join(constraints)
        return f"{description}\n\n" + "\n".join(constraints)

    def _section_budget(
        self, section: SectionDraft, section_outline: SectionOutline | None
    ) -> int | str | None:
        if section_outline is not None and section_outline.target_words is not None:
            return section_outline.target_words
        match = _BUDGET_RE.search(section.description)
        if match:
            return match.group(1)
        return None

    def _has_revision_context(self, section: SectionDraft) -> bool:
        return section.revision_count > 0 or bool(section.review_comments)

    def _writer_mode(
        self, state: AgentState, section: SectionDraft
    ) -> Literal["draft", "academic_revision", "proposal_draft", "proposal_revision"]:
        if is_research_proposal(state):
            if self._has_revision_context(section):
                return "proposal_revision"
            return "proposal_draft"
        if not self._has_revision_context(section):
            return "draft"
        return "academic_revision"

    def _draft_writer_for_mode(self, mode: str) -> DraftWriter:
        if mode == "proposal_revision":
            return self.proposal_revision_writer
        if mode == "proposal_draft":
            return self.proposal_draft_writer
        if mode == "academic_revision":
            return self.academic_revision_writer
        return self.draft_writer

    def _stream_chain_for_mode(self, mode: str) -> Any:
        if mode == "proposal_revision":
            return self.proposal_revision_chain
        if mode == "proposal_draft":
            return self.proposal_draft_chain
        if mode == "academic_revision":
            return self.academic_revision_chain
        return self.chain

    def _build_previous_sections_context(
        self,
        sections: list[SectionDraft],
        current_index: int,
        section_summaries: list[str] | None = None,
    ) -> str:
        """Build context string for sections written before the current one.

        Prefers ``section_summaries`` (pre-generated compact summaries, ~150 tokens each)
        over full section content to keep the prompt lean.
        """
        if section_summaries is not None:
            prev = [s for s in section_summaries[:current_index] if s]
            if not prev:
                return "无"
            return "\n\n---\n\n".join(prev)

        # Fallback: build from section content (legacy path for states without summaries)
        completed = [
            s for s in sections[:current_index] if s.content and s.status in ("approved", "review")
        ]
        if not completed:
            return "无"
        parts = []
        for s in completed:
            snippet = s.content[:PREVIOUS_SECTION_MAX_CHARS]
            if len(s.content) > PREVIOUS_SECTION_MAX_CHARS:
                snippet += "..."
            parts.append(f"[{s.title}]\n{snippet}")
        return "\n\n".join(parts)

    def _build_numbered_papers(
        self, papers: list[PaperMetadata], state: AgentState | None = None
    ) -> str:
        if not papers:
            return "无"
        lines = []
        for i, p in enumerate(papers, 1):
            authors_str = ", ".join(p.authors[:3])
            if len(p.authors) > 3:
                authors_str += " et al."
            abstract_snippet = p.abstract[:150] + "..." if len(p.abstract) > 150 else p.abstract
            registry = get_material_registry(state) if state is not None else None
            lines.append(
                f"[{i}] {p.title} — {authors_str}. {abstract_snippet}"
                f"{material_policy_suffix(p, registry)}"
            )
        return "\n".join(lines)

    def _build_numbered_papers_from_summaries(
        self, paper_summaries: list[str], state: AgentState | None = None
    ) -> str:
        """Use pre-built compact paper summaries from ResearcherNode (no abstract duplication)."""
        if not paper_summaries:
            return "无"
        registry = get_material_registry(state) if state is not None else None
        papers = state.get("papers", []) if state is not None else []
        return "\n".join(annotate_paper_summaries(paper_summaries, papers, registry))

    def _build_citable_sources(
        self,
        state: AgentState,
        *,
        paper_summaries: list[str] | None = None,
    ) -> str:
        citation_sources = list(state.get("citation_sources", []) or [])
        if citation_sources:
            return self._build_numbered_citation_sources(citation_sources)
        if paper_summaries:
            return self._build_numbered_papers_from_summaries(paper_summaries, state)
        return self._build_numbered_papers(state.get("papers", []), state)

    def _build_numbered_citation_sources(self, citation_sources: list[Any]) -> str:
        if not citation_sources:
            return "无"
        lines: list[str] = []
        for source in citation_sources:
            citation_id = getattr(source, "citation_id", None)
            paper = getattr(source, "paper", None)
            doi = getattr(source, "doi", None)
            stable_url = getattr(source, "stable_url", None)
            if isinstance(source, dict):
                citation_id = source.get("citation_id")
                paper = source.get("paper")
                doi = source.get("doi")
                stable_url = source.get("stable_url")
            if paper is None:
                continue
            title = (
                getattr(paper, "title", "")
                if not isinstance(paper, dict)
                else paper.get("title", "")
            )
            authors = (
                getattr(paper, "authors", [])
                if not isinstance(paper, dict)
                else paper.get("authors", [])
            )
            year = (
                getattr(paper, "year", None)
                if not isinstance(paper, dict)
                else paper.get("year")
            )
            venue = (
                getattr(paper, "venue", None)
                if not isinstance(paper, dict)
                else paper.get("venue")
            )
            authors_str = ", ".join(authors[:3]) if authors else "Unknown"
            if len(authors) > 3:
                authors_str += " et al."
            metadata = []
            if year:
                metadata.append(str(year))
            if venue:
                metadata.append(str(venue))
            if doi:
                metadata.append(f"DOI: {doi}")
            elif stable_url:
                metadata.append(str(stable_url))
            suffix = f" — {authors_str}"
            if metadata:
                suffix += ". " + ". ".join(metadata)
            lines.append(f"[{citation_id}] {title}{suffix}")
        return "\n".join(lines) if lines else "无"

    async def _move_to_next(self, state: AgentState) -> dict[str, Any]:
        return self.execution_strategy.skip_completed_section_delta(state)

    def _clean_content(self, content: str) -> str:
        return self.style_polisher.polish(content)
