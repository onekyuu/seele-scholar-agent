from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .state import AgentState, SectionOutline, SectionStyleGuidance


class StyleReference(BaseModel):
    reference_id: str
    description: str = ""
    text: str


class LocaleStylePack(BaseModel):
    locale: str
    display_name: str
    general_guidance: list[str] = Field(default_factory=list)
    paper_type_guidance: dict[str, list[str]] = Field(default_factory=dict)
    default_section_style: SectionStyleGuidance = Field(default_factory=SectionStyleGuidance)
    paper_type_section_styles: dict[str, SectionStyleGuidance] = Field(default_factory=dict)
    style_references: list[StyleReference] = Field(default_factory=list)


_LANGUAGE_TO_LOCALE = {
    "zh": "zh-CN",
    "en": "en-US",
    "ja": "ja-JP",
}


DEFAULT_STYLE_PACKS: dict[str, LocaleStylePack] = {
    "zh-CN": LocaleStylePack(
        locale="zh-CN",
        display_name="中文学术写作",
        general_guidance=[
            "使用自然、紧凑的中文学术表达，避免英文句法直译。",
            "以问题、机制、证据和判断推进段落，不用模板化口号替代分析。",
            "除非论文类型确实需要，不要套用英文 IMRaD 或 survey 固定句式。",
        ],
        paper_type_guidance={
            "literature_review": [
                "按研究问题、方法谱系或争议线索组织综述，而不是逐篇罗列文献。",
                "每一小节应说明已有研究形成了什么共识、仍留下什么分歧。",
            ],
            "theoretical": [
                "先界定概念边界，再展开理论关系和解释机制。",
                "避免把概念说明写成实验论文中的方法步骤。",
            ],
            "policy_brief": [
                "围绕问题界定、利益相关方、政策工具和实施条件推进。",
                "结论应转化为可执行的政策判断，而不是泛泛总结。",
            ],
        },
        default_section_style=SectionStyleGuidance(
            argument_mode="先提出本节要解决的问题，再给出证据和分析，最后收束到本节结论。",
            sentence_style="使用中等长度句，控制抽象名词堆叠，少用机械连接词。",
            transition_style=(
                "用概念承接或问题递进完成过渡，而不是使用 Firstly/Secondly 的直译结构。"
            ),
            forbidden_patterns=[
                "This paper/section will 的逐词直译",
                "首先/其次/最后的机械堆叠",
                "起到了重要作用、具有重要意义等空泛模板句",
            ],
            style_reference_ids=["zh_academic_review_1", "zh_academic_analysis_1"],
        ),
        style_references=[
            StyleReference(
                reference_id="zh_academic_review_1",
                description="综述型段落的中文论证节奏示例",
                text=(
                    "现有研究通常从数据规模、模型结构与任务迁移三个层面解释该问题，"
                    "但三类解释并未形成完全一致的结论。规模视角强调训练资源带来的性能增益，"
                    "结构视角更关注归纳偏置对泛化能力的影响，而迁移视角则揭示了任务语境"
                    "对模型表现的约束。由此可见，相关文献的分歧并不在于是否存在提升，"
                    "而在于提升来源及其适用边界。"
                ),
            ),
            StyleReference(
                reference_id="zh_academic_analysis_1",
                description="分析型段落的中文表达密度示例",
                text=(
                    "这一结果说明，单一指标难以完整刻画系统性能。若只观察平均准确率，"
                    "模型在少数高频类别上的优势会掩盖低频类别的错误累积；若进一步结合"
                    "稳定性与代价指标，则可以发现性能提升往往伴随推理成本上升。"
                    "因此，评价框架需要同时呈现效果、鲁棒性与资源消耗之间的权衡。"
                ),
            ),
        ],
    ),
    "ja-JP": LocaleStylePack(
        locale="ja-JP",
        display_name="日本語学術文体",
        general_guidance=[
            "日本語の学術文体として、文末表現と用語表記を一貫させる。",
            "中国語や英語の接続表現を直訳せず、日本語の論理展開に合わせる。",
            "である調を基調とし、主観的な断定や会話的な表現を避ける。",
            "段落ごとに論点、根拠、考察、次の論点への接続を明示する。",
        ],
        paper_type_guidance={
            "literature_review": [
                "研究領域、方法論、論点の相違を軸に先行研究を整理する。",
                "文献を列挙するのではなく、研究潮流と未解決課題を抽出する。",
            ],
            "theoretical": [
                "概念定義、前提条件、理論的関係、説明範囲の順に論証する。",
                "抽象概念は用語の定義と適用条件を示してから展開する。",
            ],
            "policy_brief": [
                "課題設定、利害関係者、政策手段、実施上の制約を順に検討する。",
                "結論は一般論ではなく、実行条件を伴う政策的含意として述べる。",
            ],
            "case_study": [
                "事例の選定理由、分析視点、観察結果、一般化可能性を明示する。",
                "単なる事例紹介に留めず、理論的または実務的含意へ接続する。",
            ],
        },
        default_section_style=SectionStyleGuidance(
            argument_mode="問題設定、根拠、考察、含意の順に段落を展開する。",
            sentence_style="である調を基本とし、冗長な名詞句と直訳調を避ける。",
            transition_style="前段落の論点を受けて、次の検討課題を示す形で接続する。",
            forbidden_patterns=[
                "中国語式の接続語の直訳",
                "英語構文をそのまま写した長い修飾句",
            ],
            style_reference_ids=["ja_academic_review_1", "ja_academic_analysis_1"],
        ),
        paper_type_section_styles={
            "literature_review": SectionStyleGuidance(
                argument_mode=(
                    "先行研究の共通点と相違点を整理し、研究上の空白を示して論点を収束"
                    "させる。"
                ),
                sentence_style="文献名の列挙を避け、研究潮流を説明する文を中心に構成する。",
                transition_style="前節で整理した知見から、次に検討すべき課題を導く。",
                forbidden_patterns=[
                    "論文ごとの単純な要約列挙",
                    "Firstly/Secondly の直訳的な構成",
                    "中国語式の「一方で／さらに」の機械的反復",
                ],
                style_reference_ids=["ja_academic_review_1"],
            ),
            "theoretical": SectionStyleGuidance(
                argument_mode="概念定義から理論関係を導き、説明可能な範囲と限界を示す。",
                sentence_style="抽象概念を連続させず、定義と含意を対応させて記述する。",
                transition_style="概念間の関係を明示しながら次の論点へ移る。",
                forbidden_patterns=[
                    "方法・実験・結果という実証論文の定型構成",
                    "英語の関係節を写した長い連体修飾",
                ],
                style_reference_ids=["ja_academic_analysis_1"],
            ),
            "policy_brief": SectionStyleGuidance(
                argument_mode="課題、根拠、政策選択肢、実施条件、含意の順に論じる。",
                sentence_style="過度な断定を避け、条件付きの政策判断として記述する。",
                transition_style="前段落の制約条件を受け、次の政策選択肢を提示する。",
                forbidden_patterns=[
                    "重要であるだけで終わる一般論",
                    "根拠のない提言",
                ],
                style_reference_ids=["ja_academic_analysis_1"],
            ),
        },
        style_references=[
            StyleReference(
                reference_id="ja_academic_review_1",
                description="レビュー論文における研究潮流整理の例",
                text=(
                    "先行研究は、モデル規模、学習データ、評価指標という三つの観点から"
                    "この問題を検討してきた。ただし、各観点が示す説明は必ずしも同じ"
                    "方向を向いていない。モデル規模に着目する研究は性能向上の量的側面を"
                    "強調する一方、評価指標に注目する研究は、性能の上昇が特定の条件に"
                    "依存することを示している。したがって、本領域の課題は性能差の有無"
                    "ではなく、その差がどの条件で生じるのかを明らかにする点にある。"
                ),
            ),
            StyleReference(
                reference_id="ja_academic_analysis_1",
                description="分析型段落における根拠と含意の接続例",
                text=(
                    "この結果は、単一の評価指標だけでは対象システムの性質を十分に"
                    "捉えられないことを示している。平均的な性能だけを確認すると、"
                    "特定の条件下で生じる誤差の偏りが見えにくくなる。これに対して、"
                    "安定性や計算コストを併せて検討すれば、性能向上がどのような代償を"
                    "伴うのかを評価できる。"
                ),
            ),
        ],
    ),
    "en-US": LocaleStylePack(
        locale="en-US",
        display_name="English academic prose",
        general_guidance=[
            "Use concise academic prose with explicit claims, evidence, and analysis.",
            "Match the structure to the selected paper type instead of forcing IMRaD.",
        ],
        default_section_style=SectionStyleGuidance(
            argument_mode="State the section claim, support it with evidence, then analyze scope.",
            sentence_style="Prefer precise, active academic prose over inflated abstractions.",
            transition_style="Use conceptual transitions that show how the next point follows.",
            forbidden_patterns=["Generic filler sentences without evidence"],
            style_reference_ids=[],
        ),
    ),
}


def resolve_writing_locale(state: AgentState) -> str:
    explicit_locale = str(state.get("writing_locale") or "").strip()
    if explicit_locale:
        if explicit_locale in _LANGUAGE_TO_LOCALE:
            return _LANGUAGE_TO_LOCALE[explicit_locale]
        return explicit_locale
    return _LANGUAGE_TO_LOCALE.get(str(state.get("language", "zh")), "zh-CN")


def resolve_style_pack(state: AgentState) -> LocaleStylePack | None:
    locale = resolve_writing_locale(state)
    base_pack = DEFAULT_STYLE_PACKS.get(locale)
    raw_override = state.get("style_pack_override")
    if raw_override:
        override_pack = _parse_override_pack(raw_override, locale)
        if override_pack is not None:
            return override_pack
    return base_pack


def build_planner_style_context(
    state: AgentState, paper_type: str = "auto", structure_pattern: str = "auto"
) -> str:
    pack = resolve_style_pack(state)
    if pack is None:
        return "No locale-specific style pack is configured. Apply only general academic style."

    lines = _base_style_context_lines(state, pack, paper_type)
    lines.extend(
        [
            "",
            "Planner requirements:",
            "- For each section, add section_style with argument_mode, sentence_style, "
            "transition_style, forbidden_patterns, and style_reference_ids.",
            "- Select section_style according to writing_locale, paper_type, and "
            "structure_pattern.",
            "- Do not apply style guidance from another language.",
            f"- Current structure_pattern: {structure_pattern}",
        ]
    )
    return "\n".join(lines)


def build_writer_style_context(
    state: AgentState, section_outline: SectionOutline | None = None
) -> str:
    pack = resolve_style_pack(state)
    if pack is None:
        return "No locale-specific style pack is configured. Apply only general academic style."

    outline = state.get("outline")
    paper_type = str(state.get("paper_type") or getattr(outline, "paper_type", "auto") or "auto")
    lines = _base_style_context_lines(state, pack, paper_type)
    section_style = (
        section_outline.section_style
        if section_outline is not None
        else _style_for_paper_type(pack, paper_type)
    )
    lines.extend(["", "Current section style:", *_section_style_lines(section_style)])
    return "\n".join(lines)


def _base_style_context_lines(
    state: AgentState, pack: LocaleStylePack, paper_type: str
) -> list[str]:
    lines = [
        f"Writing locale: {pack.locale} ({pack.display_name})",
        f"Style profile: {state.get('style_profile') or 'default'}",
        f"Paper type: {paper_type}",
    ]
    if pack.general_guidance:
        lines.append("General guidance:")
        lines.extend(f"- {item}" for item in pack.general_guidance)

    type_guidance = pack.paper_type_guidance.get(paper_type)
    if type_guidance:
        lines.append("Paper-type guidance:")
        lines.extend(f"- {item}" for item in type_guidance)

    term_glossary = state.get("term_glossary") or {}
    if term_glossary:
        lines.append("Terminology glossary:")
        lines.extend(f"- {source}: {target}" for source, target in term_glossary.items())

    if pack.style_references:
        lines.append(
            "Style references are original short examples. Imitate expression density and "
            "sentence rhythm only; do not reuse their content."
        )
        for reference in pack.style_references[:2]:
            lines.append(
                f"[{reference.reference_id}] {reference.description}: {reference.text}"
            )
    return lines


def _style_for_paper_type(pack: LocaleStylePack, paper_type: str) -> SectionStyleGuidance:
    return pack.paper_type_section_styles.get(paper_type) or pack.default_section_style


def _section_style_lines(section_style: SectionStyleGuidance) -> list[str]:
    lines = [
        f"- argument_mode: {section_style.argument_mode or 'default'}",
        f"- sentence_style: {section_style.sentence_style or 'default'}",
        f"- transition_style: {section_style.transition_style or 'default'}",
    ]
    if section_style.forbidden_patterns:
        lines.append("- forbidden_patterns: " + "; ".join(section_style.forbidden_patterns))
    if section_style.style_reference_ids:
        lines.append("- style_reference_ids: " + "; ".join(section_style.style_reference_ids))
    return lines


def _parse_override_pack(raw_override: dict[str, Any], locale: str) -> LocaleStylePack | None:
    candidate = raw_override
    if "locales" in raw_override:
        raw_locales = raw_override.get("locales")
        if isinstance(raw_locales, dict):
            locale_candidate = raw_locales.get(locale)
            if isinstance(locale_candidate, dict):
                candidate = locale_candidate
    if not isinstance(candidate, dict):
        return None
    data = {"locale": locale, "display_name": locale, **candidate}
    try:
        return LocaleStylePack.model_validate(data)
    except ValidationError:
        return None
