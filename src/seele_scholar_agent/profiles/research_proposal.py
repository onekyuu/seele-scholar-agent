import re
from typing import Any

from ..document_profile import (
    is_proposal_plan_sentence,
    is_schedule_section,
    missing_proposal_core_tasks,
    missing_schedule_phases,
)
from ..state import (
    OutlineStructure,
    QualityIssue,
    ReviewIssue,
    ReviewResult,
    SectionEvidencePlan,
    SectionOutline,
)
from .base import (
    PROFILE_DRAFT_MODE,
    PROFILE_REVISION_MODE,
    ClaimSourceAuditCase,
    ProfileWriterPrompts,
    ReviewIssueCategory,
    WriterMode,
)

RESEARCH_PROPOSAL_PROFILE_NAME = "research_proposal"

PROPOSAL_REVISION_USER_PROMPT = """研究主题：{topic}

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

PROPOSAL_DRAFT_USER_PROMPT = """研究主题：{topic}

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

_PROPOSAL_RESULT_MARKERS = (
    "results show",
    "result shows",
    "experiment showed",
    "experiments showed",
    "we found",
    "we achieved",
    "achieved",
    "outperformed",
    "improved by",
    "結果は",
    "結果が",
    "実験結果",
    "示した",
    "達成した",
    "上回った",
    "改善した",
    "结果显示",
    "实验表明",
    "我们发现",
)


class ResearchProposalProfile:
    name = RESEARCH_PROPOSAL_PROFILE_NAME
    allow_empty_references = True
    skip_auto_finalizer = True
    uses_profile_outline_quality = True
    uses_strict_academic_integrity = False
    review_document_type = RESEARCH_PROPOSAL_PROFILE_NAME
    uses_specialized_review_policy = True

    def effective_paper_type(self, requested: str) -> str:
        return RESEARCH_PROPOSAL_PROFILE_NAME if requested == "auto" else requested

    def effective_structure_pattern(self, requested: str) -> str:
        return RESEARCH_PROPOSAL_PROFILE_NAME if requested == "auto" else requested

    def default_outline(self, topic: str, lang: str) -> dict[str, Any] | None:
        return default_proposal_outline(topic)

    def normalize_outline(
        self, outline: OutlineStructure, topic: str
    ) -> OutlineStructure:
        return normalize_proposal_outline(outline, topic)

    def planner_context_suffix(self, target_word_count: str) -> str:
        proposal_lines = [
            "Research proposal requirements:",
            "- Treat this as a Japanese graduate-school research proposal, not a paper.",
            "- Plan a lightweight 4-5 chapter structure for a complete 2000-3000 "
            "Japanese-character document.",
            "- Prefer separate, single-task titles such as 研究背景, 先行研究と課題, "
            "研究目的, 研究方法・計画, 期待される成果.",
            "- Avoid heavy dual-task titles. If a compound title is used, its description "
            "must say the second part only needs overview-level coverage, not a full "
            "paper-style subsection.",
            "- Include motivation, prior-work gap, research purpose/questions, method, "
            "feasibility/plan, and expected outcomes at application-review density.",
            "- Only explicit schedule/timeline sections must cover 1年次前期, 1年次後期, "
            "2年次前期, and 2年次後期.",
            "- Do not require citations for the applicant's own plan, intended work, "
            "timeline, deliverables, or future evaluation.",
            "- Use citations only for prior-work/background claims.",
            f"- Total target length: {target_word_count}. Allocate target_words per section.",
        ]
        return "\n".join(proposal_lines)

    def writer_mode(self, has_revision_context: bool) -> WriterMode:
        return PROFILE_REVISION_MODE if has_revision_context else PROFILE_DRAFT_MODE

    def writer_prompts(self, prompts: Any) -> ProfileWriterPrompts | None:
        return ProfileWriterPrompts(
            draft_user_prompt=prompts.proposal_writer_user_prompt
            or PROPOSAL_DRAFT_USER_PROMPT,
            revision_user_prompt=prompts.proposal_revision_user_prompt
            or PROPOSAL_REVISION_USER_PROMPT,
        )

    def review_policy_text(self) -> str:
        return (
            "Review as a Japanese graduate-school research proposal application, not as "
            "an academic paper body section. Only block for off-topic content, missing "
            "title-core task, truncation/incomplete sentences, impossible-to-judge "
            "purpose/method/plan feasibility, severe factual error, misleading citation, "
            "or enumeration/structure breakage. Treat most missing citations, citation "
            "mismatches, and unsupported claims as warnings unless they mislead the "
            "proposal."
        )

    def missing_core_tasks(self, section_title: str, content: str) -> list[str]:
        return missing_proposal_core_tasks(section_title, content)

    def should_defer_claim(
        self, claim_text: str, citation_numbers: tuple[int, ...], section_title: str
    ) -> bool:
        return not citation_numbers and is_proposal_plan_sentence(claim_text, section_title)

    def citation_alignment_uses_cited_context(self) -> bool:
        return True

    def citation_review_category(self) -> ReviewIssueCategory:
        return "citation_warning"

    def should_emit_claim_source_review_issue(self, audit_case: ClaimSourceAuditCase) -> bool:
        return audit_case != "unsupported_binding"

    def claim_source_quality_issue(
        self,
        quality_issue: QualityIssue,
        *,
        audit_source: str,
        binding_diagnostics: dict[str, Any] | None = None,
    ) -> QualityIssue:
        details = {
            **quality_issue.details,
            "audit_source": audit_source,
            "deferred": True,
        }
        if binding_diagnostics is not None:
            details["binding_diagnostics"] = binding_diagnostics
        return quality_issue.model_copy(
            update={
                "severity": "warning",
                "details": details,
            }
        )

    def include_paragraph_structure_check(self) -> bool:
        return False

    def empty_reference_issue(self) -> QualityIssue | None:
        return QualityIssue(
            code="PROPOSAL_NO_INLINE_CITATIONS",
            message=(
                "No inline citations were found. For research proposals this is "
                "allowed, but prior-work/background sections should cite sources "
                "when they make literature claims."
            ),
            severity="warning",
            location="references",
            blocking=False,
        )

    def structural_review_issues(
        self, section_id: str, section_title: str, content: str
    ) -> tuple[list[ReviewIssue], list[QualityIssue]]:
        issues: list[ReviewIssue] = []
        quality_issues: list[QualityIssue] = []

        if is_schedule_section(section_title):
            missing = missing_schedule_phases(content)
            if missing:
                issue = ReviewIssue(
                    type="format_issue",
                    description=(
                        "Research proposal schedule is incomplete; missing phases: "
                        + ", ".join(missing)
                    ),
                    suggestion=(
                        "Revise the schedule to cover 1年次前期, 1年次後期, 2年次前期, "
                        "and 2年次後期, with tasks and deliverables for each phase."
                    ),
                    location=section_title,
                    blocking=True,
                    category="blocking",
                )
                quality_issue = QualityIssue(
                    code="PROPOSAL_SCHEDULE_PHASES_MISSING",
                    message=issue.description,
                    severity="blocking",
                    location=section_title,
                    blocking=True,
                    details={
                        "section_id": section_id,
                        "audit_source": "structural",
                        "missing_phases": missing,
                    },
                )
                issues.append(issue)
                quality_issues.append(quality_issue)

        core_issues, core_quality_issues = self._core_structure_issues(
            section_id, section_title, content
        )
        return [*issues, *core_issues], [*quality_issues, *core_quality_issues]

    def apply_review_policy(
        self, review: ReviewResult, quality_issues: list[QualityIssue]
    ) -> tuple[ReviewResult, list[QualityIssue]]:
        normalized_issues = [self._normalize_review_issue(issue) for issue in review.issues]
        has_blocking_issue = any(issue.blocking for issue in normalized_issues) or any(
            issue.blocking or issue.severity == "blocking" for issue in quality_issues
        )

        approved = review.approved
        if has_blocking_issue:
            approved = False
        elif review.score >= 7:
            approved = True

        return review.model_copy(
            update={"approved": approved, "issues": normalized_issues}
        ), quality_issues

    def review_diagnostic_fields(self, section_title: str, content: str) -> dict[str, Any]:
        return {
            "proposal_profile": True,
            "reviewer_mode": "proposal_review",
            "missing_core_tasks": (
                missing_proposal_core_tasks(section_title, content) if section_title else []
            ),
        }

    def _core_structure_issues(
        self, section_id: str, section_title: str, content: str
    ) -> tuple[list[ReviewIssue], list[QualityIssue]]:
        issues: list[ReviewIssue] = []
        quality_issues: list[QualityIssue] = []
        stripped = content.strip()

        if not stripped:
            issue = self._blocking_review_issue(
                "Proposal section is empty.",
                "Write a compact section that addresses the title's core task.",
                section_title,
            )
            return [issue], [self._quality_issue(issue, section_id)]

        if self._looks_truncated(stripped):
            issue = self._blocking_review_issue(
                "Proposal section appears truncated or ends with an incomplete sentence.",
                "Complete the final sentence while staying within the section budget.",
                section_title,
            )
            issues.append(issue)
            quality_issues.append(self._quality_issue(issue, section_id))

        enumeration_issue = self._enumeration_issue(section_title, stripped)
        if enumeration_issue is not None:
            issues.append(enumeration_issue)
            quality_issues.append(self._quality_issue(enumeration_issue, section_id))

        missing_core_tasks = missing_proposal_core_tasks(section_title, stripped)
        if missing_core_tasks:
            issue = self._blocking_review_issue(
                "Proposal section is missing title-core task(s): "
                + ", ".join(missing_core_tasks),
                (
                    "Add overview-level coverage for the missing task(s); do not expand "
                    "into a full paper-style subsection."
                ),
                section_title,
            )
            issues.append(issue)
            quality_issues.append(self._quality_issue(issue, section_id))

        return issues, quality_issues

    def _blocking_review_issue(
        self, description: str, suggestion: str, location: str
    ) -> ReviewIssue:
        return ReviewIssue(
            type="format_issue",
            description=description,
            suggestion=suggestion,
            location=location,
            blocking=True,
            category="blocking",
        )

    def _quality_issue(self, review_issue: ReviewIssue, section_id: str) -> QualityIssue:
        return QualityIssue(
            code=self._quality_code(review_issue.description),
            message=review_issue.description,
            severity="blocking",
            location=review_issue.location,
            blocking=True,
            details={
                "section_id": section_id,
                "audit_source": "proposal_structure",
                "category": review_issue.category,
            },
        )

    def _quality_code(self, description: str) -> str:
        if "truncated" in description or "incomplete sentence" in description:
            return "PROPOSAL_SECTION_TRUNCATED"
        if "missing title-core task" in description:
            return "PROPOSAL_CORE_TASK_MISSING"
        if "enumeration" in description or "declares" in description:
            return "PROPOSAL_ENUMERATION_INCONSISTENT"
        return "PROPOSAL_SECTION_STRUCTURAL_BLOCK"

    def _looks_truncated(self, content: str) -> bool:
        if content.endswith(("。", ".", "!", "?", "！", "？", "」", "』", "）", ")", "】")):
            return False
        return True

    def _enumeration_issue(
        self, section_title: str, content: str
    ) -> ReviewIssue | None:
        declared_count = self._declared_enumeration_count(content)
        if declared_count is None:
            return None
        actual_count = self._actual_enumeration_count(content)
        if actual_count >= declared_count:
            return None
        return self._blocking_review_issue(
            (
                f"Section declares {declared_count} points/stages but only "
                f"{actual_count} are explicitly developed."
            ),
            (
                "Make the declared number match the actual text, or add the missing "
                "point/stage compactly."
            ),
            section_title,
        )

    def _declared_enumeration_count(self, content: str) -> int | None:
        patterns: tuple[tuple[str, int], ...] = (
            (r"(?:三点|3点|三つ|三段階|3段階|three (?:points|stages))", 3),
            (r"(?:二点|2点|二つ|二段階|2段階|two (?:points|stages))", 2),
        )
        for pattern, count in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return count
        return None

    def _actual_enumeration_count(self, content: str) -> int:
        markers = re.findall(
            r"(?:第[一二三四五](?:に|段階|ステップ)|[一二三四五]つ目|[（(]?[1-5][）).、]|"
            r"\b(?:first|second|third|fourth|fifth)\b)",
            content,
            re.IGNORECASE,
        )
        return len(set(markers))

    def _normalize_review_issue(self, issue: ReviewIssue) -> ReviewIssue:
        if issue.blocking:
            return issue.model_copy(update={"category": "blocking"})
        if issue.type in {"missing_citation", "citation_mismatch"}:
            return issue.model_copy(
                update={"blocking": False, "category": "citation_warning"}
            )
        if issue.type == "factual_error":
            return issue.model_copy(update={"blocking": True, "category": "blocking"})
        if issue.type == "format_issue":
            return issue.model_copy(update={"blocking": False, "category": "format"})
        return issue.model_copy(update={"blocking": False, "category": "content_quality"})

    def outline_section_issues(
        self, section: SectionOutline, *, is_last: bool
    ) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        location = f"outline.sections.{section.order}"
        if not section.purpose.strip():
            issues.append(
                _blocking_issue(
                    "OUTLINE_MISSING_PURPOSE",
                    f"Section '{section.title}' is missing purpose.",
                    location,
                )
            )
        if not is_last and not section.transition_to_next.strip():
            issues.append(
                QualityIssue(
                    code="OUTLINE_MISSING_TRANSITION",
                    message=f"Section '{section.title}' is missing transition_to_next.",
                    severity="warning",
                    location=location,
                    blocking=False,
                )
            )
        if section.target_words is None:
            issues.append(
                QualityIssue(
                    code="OUTLINE_MISSING_TARGET_WORDS",
                    message=f"Section '{section.title}' has no proposal length budget.",
                    severity="warning",
                    location=location,
                    blocking=False,
                )
            )
        return issues

    def outline_structure_issues(self, outline: OutlineStructure) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        sections = sorted(outline.sections, key=lambda item: item.order)
        if len(sections) < 4 or len(sections) > 5:
            issues.append(
                QualityIssue(
                    code="PROPOSAL_SECTION_COUNT_OUT_OF_RANGE",
                    message="Research proposal outline should usually have 4-5 sections.",
                    severity="warning",
                    location="outline.sections",
                    blocking=False,
                    details={"section_count": len(sections)},
                )
            )

        schedule_sections = [section for section in sections if is_schedule_section(section.title)]
        plan_sections = [
            section
            for section in sections
            if "計画" in section.title or "plan" in section.title.casefold()
        ]
        if not schedule_sections and not plan_sections:
            issues.append(
                QualityIssue(
                    code="PROPOSAL_PLAN_SECTION_MISSING",
                    message=(
                        "Research proposal outline should include method/plan or "
                        "schedule information for feasibility review."
                    ),
                    severity="warning",
                    location="outline.sections",
                    blocking=False,
                )
            )
            return issues
        if not schedule_sections:
            return issues

        schedule = schedule_sections[0]
        schedule_text = "\n".join(
            [
                schedule.title,
                schedule.description,
                schedule.content_summary,
                " ".join(schedule.key_points),
            ]
        )
        missing = missing_schedule_phases(schedule_text)
        if missing:
            issues.append(
                _blocking_issue(
                    "PROPOSAL_SCHEDULE_PHASES_MISSING",
                    "Schedule outline is missing phases: " + ", ".join(missing),
                    f"outline.sections.{schedule.order}",
                    details={"missing_phases": missing},
                )
            )
        return issues

    def skip_methodology_audit(self, content: str) -> bool:
        lowered = content.casefold()
        return not any(marker.casefold() in lowered for marker in _PROPOSAL_RESULT_MARKERS)


def default_proposal_outline(topic: str) -> dict[str, Any]:
    sections = [
        {
            "title": "研究背景",
            "description": (
                "申請審査に必要な背景、問題意識、研究テーマとの接続を"
                "約350-450字で簡潔に述べる。"
            ),
            "order": 1,
            "purpose": "研究計画書の問題意識と研究動機を明確にする。",
            "content_summary": "背景、問題意識、申請者固有の関心を簡潔に説明する。",
            "target_words": 400,
            "key_points": ["研究背景", "問題意識", "研究テーマとの接続"],
            "target_claims": [],
            "key_sources": [],
            "citation_plan": [],
            "evidence_gaps": [],
            "transition_to_next": "この問題意識を受けて研究目的を定義する。",
            "section_style": {},
            "suggested_figures": [],
        },
        {
            "title": "先行研究と課題",
            "description": (
                "主要な先行研究の位置づけと残された課題を約400-500字で述べる。"
                "論文の関連研究章のように網羅せず、申請審査に必要な情報密度に絞る。"
            ),
            "order": 2,
            "purpose": "本研究が取り組む課題の位置づけを示す。",
            "content_summary": "先行研究の要点、限界、未解決課題を申請書向けに整理する。",
            "target_words": 450,
            "key_points": ["先行研究", "残された課題", "本研究の位置づけ"],
            "target_claims": [],
            "key_sources": [],
            "citation_plan": [],
            "evidence_gaps": [],
            "transition_to_next": "課題を受けて研究目的へ接続する。",
            "section_style": {},
            "suggested_figures": [],
        },
        {
            "title": "研究目的",
            "description": (
                "本研究で明らかにしたいこと、研究対象、問いを約350-450字で"
                "具体化する。期待成果は概要に留め、方法章へ接続する。"
            ),
            "order": 3,
            "purpose": "本研究が何を明らかにするかを示す。",
            "content_summary": "研究目的、研究質問、対象範囲を簡潔に述べる。",
            "target_words": 400,
            "key_points": ["研究目的", "研究質問", "対象範囲"],
            "target_claims": [],
            "key_sources": [],
            "citation_plan": [],
            "evidence_gaps": [],
            "transition_to_next": "目的達成のための方法と計画へ接続する。",
            "section_style": {},
            "suggested_figures": [],
        },
        {
            "title": "研究方法・計画",
            "description": (
                "使用する資料・ツール・方法、分析や検証の進め方、修士段階での"
                "実行可能性を約500-600字で述べる。複合タイトルの「計画」は"
                "概要級でよく、論文の方法章のような実験 protocol、詳細な変数設計、"
                "統計検定までは要求しない。"
            ),
            "order": 4,
            "purpose": "研究計画の方法と修士段階での実行可能性を示す。",
            "content_summary": "資料、方法、検証方針、概略スケジュールを申請書向けに述べる。",
            "target_words": 550,
            "key_points": ["資料・ツール", "分析・検証方針", "修士段階での実行可能性"],
            "target_claims": [],
            "key_sources": [],
            "citation_plan": [],
            "evidence_gaps": [],
            "transition_to_next": "方法と計画を踏まえて期待される成果へ接続する。",
            "section_style": {},
            "suggested_figures": [],
        },
        {
            "title": "期待される成果",
            "description": (
                "期待される成果、申請上の価値、将来展望を約250-350字で述べる。"
                "1-2個の具体的貢献でよく、論文の contribution section のような"
                "網羅的展開は不要。"
            ),
            "order": 5,
            "purpose": "研究の意義と進学後の展望を締めくくる。",
            "content_summary": "成果、申請価値、将来展望を簡潔にまとめる。",
            "target_words": 300,
            "key_points": ["期待成果", "申請価値"],
            "target_claims": [],
            "key_sources": [],
            "citation_plan": [],
            "evidence_gaps": [],
            "transition_to_next": "",
            "section_style": {},
            "suggested_figures": [],
        },
    ]
    return {
        "title": f"{topic}に関する研究計画書",
        "abstract": "",
        "sections": sections,
        "keywords": [topic],
        "paper_type": RESEARCH_PROPOSAL_PROFILE_NAME,
        "structure_pattern": RESEARCH_PROPOSAL_PROFILE_NAME,
        "rationale": "Fallback research proposal outline for Japanese graduate admission.",
        "evidence_map": [],
    }


def normalize_proposal_outline(outline: OutlineStructure, topic: str) -> OutlineStructure:
    if not outline.sections:
        return _build_outline_structure(default_proposal_outline(topic))

    sections = [
        _with_proposal_description_guard(section)
        for section in sorted(outline.sections, key=lambda section: section.order)
    ]
    evidence_map = [
        SectionEvidencePlan(
            section_title=section.title,
            target_claims=section.target_claims,
            key_sources=section.key_sources,
            evidence_gaps=section.evidence_gaps,
            citation_plan=section.citation_plan,
        )
        for section in sections
    ]
    return outline.model_copy(
        update={
            "paper_type": RESEARCH_PROPOSAL_PROFILE_NAME,
            "structure_pattern": RESEARCH_PROPOSAL_PROFILE_NAME,
            "sections": sections,
            "evidence_map": evidence_map,
        }
    )


def _build_outline_structure(raw: dict[str, Any]) -> OutlineStructure:
    sections = [
        SectionOutline(
            title=str(section.get("title", f"Section {index}")),
            description=str(section.get("description", "")),
            order=int(section.get("order", index)),
            key_points=list(section.get("key_points", [])),
            suggested_figures=list(section.get("suggested_figures", [])),
            purpose=str(section.get("purpose", "")),
            content_summary=str(section.get("content_summary", "")),
            target_words=section.get("target_words"),
            target_claims=list(section.get("target_claims", [])),
            key_sources=list(section.get("key_sources", [])),
            evidence_gaps=list(section.get("evidence_gaps", [])),
            citation_plan=list(section.get("citation_plan", [])),
            transition_to_next=str(section.get("transition_to_next", "")),
        )
        for index, section in enumerate(raw.get("sections", []), 1)
        if isinstance(section, dict)
    ]
    return OutlineStructure(
        title=str(raw.get("title", "")),
        abstract=str(raw.get("abstract", "")),
        sections=sections,
        keywords=list(raw.get("keywords", [])),
        paper_type=RESEARCH_PROPOSAL_PROFILE_NAME,
        structure_pattern=RESEARCH_PROPOSAL_PROFILE_NAME,
        rationale=str(raw.get("rationale", "")),
        evidence_map=[],
    )


def _with_proposal_description_guard(section: SectionOutline) -> SectionOutline:
    compound_markers = ("・", "/", "／", "&", " and ", "、")
    title_lower = section.title.casefold()
    if not any(marker in title_lower for marker in compound_markers):
        return section
    guard = (
        " 複合タイトルの場合、第二要素は申請審査に必要な概要級のカバーでよく、"
        "論文本文の独立小節のような完全展開は要求しない。"
    )
    if "概要級" in section.description or "overview-level" in section.description:
        return section
    return section.model_copy(update={"description": section.description + guard})


def _blocking_issue(
    code: str,
    message: str,
    location: str,
    *,
    details: dict[str, Any] | None = None,
) -> QualityIssue:
    return QualityIssue(
        code=code,
        message=message,
        severity="blocking",
        location=location,
        blocking=True,
        details=details or {},
    )
