import re
from collections.abc import AsyncIterator
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig, RAGRetrieverFunc
from ..i18n import t
from ..logging import get_logger
from ..state import (
    AgentState,
    ClaimEvidenceBinding,
    DocumentChunk,
    EvidencePacket,
    PaperMetadata,
    SectionDraft,
)
from . import (
    CITATION_PATTERN,
    PREVIOUS_SECTION_MAX_CHARS,
    SECTION_SUMMARY_MAX_CHARS,
    NodeStreamEvent,
    _stream_llm_text,
    invoke_with_retry,
)
from .claim_audit import RuleBasedClaimExtractor

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


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
                packet = self._select_packet(citation_number, claim_text, papers, evidence_packets)
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
    ) -> EvidencePacket | None:
        if not evidence_packets:
            return None

        source_paper_id = None
        if 1 <= citation_number <= len(papers):
            source_paper_id = papers[citation_number - 1].paper_id

        candidates = [
            packet for packet in evidence_packets if packet.source_paper_id == source_paper_id
        ]
        if not candidates:
            candidates = evidence_packets

        return max(
            candidates,
            key=lambda packet: (
                _token_overlap_score(claim_text, packet.quote),
                packet.relevance_score,
            ),
        )

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
        if score >= 0.35:
            return "supported"
        if score >= 0.15:
            return "weak"
        return "unsupported"


class WriterNode:
    def __init__(
        self, llm: ChatOpenAI, prompts: PromptsConfig, rag_retriever: RAGRetrieverFunc | None = None
    ):
        self.llm = llm
        self.prompts = prompts
        self.rag_retriever = rag_retriever
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", prompts.writer_system_prompt), ("user", prompts.writer_user_prompt)]
        )
        self.chain = self.prompt | self.llm
        self.draft_writer = DraftWriter(self.chain)
        self.citation_binder = CitationBinder()
        self.style_polisher = StylePolisher()

    async def write(self, state: AgentState) -> dict[str, Any]:
        sections = state["sections"]
        current_index = state["current_section_index"]
        lang = state.get("language", "zh")

        if current_index >= len(sections):
            return {
                "status": "completed",
                "sections_completed": state.get("sections_completed", []),
            }

        section = sections[current_index]

        if section.status == "approved":
            return await self._move_to_next(state)

        logger.info("writing section", title=section.title, language=lang)

        evidence_packets, new_evidence_packets = await self._collect_evidence_packets(
            state, section
        )
        rag_context = self._build_rag_context(evidence_packets)

        outline_json = self._build_outline_json(state.get("outline"))
        review_comments = self._build_review_comments(section)

        section_summaries: list[str] = list(state.get("section_summaries") or [])
        previous_sections = self._build_previous_sections_context(
            sections, current_index, section_summaries
        )

        paper_summaries: list[str] = state.get("paper_summaries") or []
        numbered_papers = (
            self._build_numbered_papers_from_summaries(paper_summaries)
            if paper_summaries
            else self._build_numbered_papers(state.get("papers", []))
        )

        try:
            raw_content = await self.draft_writer.draft(
                self._build_draft_input(
                    state,
                    section,
                    lang,
                    outline_json,
                    previous_sections,
                    numbered_papers,
                    rag_context,
                    review_comments,
                )
            )
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

        updated_summaries = list(state.get("section_summaries") or [])
        while len(updated_summaries) <= current_index:
            updated_summaries.append("")
        updated_summaries[current_index] = _generate_section_summary(section.title, content)

        result: dict[str, Any] = {
            "sections": updated_sections,
            "section_summaries": updated_summaries,
            "claim_evidence_bindings": claim_evidence_bindings,
            "status": "reviewing",
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
                    "sections_completed": state.get("sections_completed", []),
                },
            )
            return

        section = sections[current_index]

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

        input_data = {
            "topic": state["topic"],
            "language": t(lang, "language_name"),
            "section_title": section.title,
            "section_description": section.description,
            "suggested_figures": self._build_suggested_figures(section, state),
            "outline_json": self._build_outline_json(state.get("outline")),
            "previous_sections": self._build_previous_sections_context(
                sections, current_index, _section_summaries
            ),
            "numbered_papers": (
                self._build_numbered_papers_from_summaries(_paper_summaries)
                if _paper_summaries
                else self._build_numbered_papers(state.get("papers", []))
            ),
            "rag_context": rag_context,
            "review_comments": self._build_review_comments(section),
        }

        full_text = ""
        async for event in _stream_llm_text(self.chain, input_data):
            full_text += event.get("token", "")
            yield event

        content = self._clean_content(full_text)
        claim_evidence_bindings = self.citation_binder.bind(
            section, content, state.get("papers", []), evidence_packets
        )

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
            "claim_evidence_bindings": claim_evidence_bindings,
            "status": "reviewing",
        }
        if new_evidence_packets:
            result["evidence_packets"] = new_evidence_packets

        yield NodeStreamEvent(type="result", result=result)

    def _build_draft_input(
        self,
        state: AgentState,
        section: SectionDraft,
        lang: str,
        outline_json: str,
        previous_sections: str,
        numbered_papers: str,
        rag_context: str,
        review_comments: str,
    ) -> dict[str, Any]:
        return {
            "topic": state["topic"],
            "language": t(lang, "language_name"),
            "section_title": section.title,
            "section_description": section.description,
            "suggested_figures": self._build_suggested_figures(section, state),
            "outline_json": outline_json,
            "previous_sections": previous_sections,
            "numbered_papers": numbered_papers,
            "rag_context": rag_context,
            "review_comments": review_comments,
        }

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

    def _build_numbered_papers(self, papers: list[PaperMetadata]) -> str:
        if not papers:
            return "无"
        lines = []
        for i, p in enumerate(papers, 1):
            authors_str = ", ".join(p.authors[:3])
            if len(p.authors) > 3:
                authors_str += " et al."
            abstract_snippet = p.abstract[:150] + "..." if len(p.abstract) > 150 else p.abstract
            lines.append(f"[{i}] {p.title} — {authors_str}. {abstract_snippet}")
        return "\n".join(lines)

    def _build_numbered_papers_from_summaries(self, paper_summaries: list[str]) -> str:
        """Use pre-built compact paper summaries from ResearcherNode (no abstract duplication)."""
        if not paper_summaries:
            return "无"
        return "\n".join(paper_summaries)

    async def _move_to_next(self, state: AgentState) -> dict[str, Any]:
        sections = state["sections"]
        index = state["current_section_index"]
        completed = state.get("sections_completed", [])
        completed.append(sections[index].title)

        if index + 1 >= len(sections):
            return {"sections_completed": completed, "status": "completed"}

        return {
            "sections_completed": completed,
            "current_section_index": index + 1,
            "status": "writing",
        }

    def _clean_content(self, content: str) -> str:
        return self.style_polisher.polish(content)
