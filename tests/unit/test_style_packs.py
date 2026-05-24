from seele_scholar_agent.state import AgentState
from seele_scholar_agent.style_packs import (
    build_planner_style_context,
    build_writer_style_context,
    resolve_writing_locale,
)


def test_resolve_writing_locale_uses_explicit_locale(base_state: AgentState):
    state = AgentState(**{**base_state, "language": "ja", "writing_locale": "zh-CN"})

    assert resolve_writing_locale(state) == "zh-CN"


def test_resolve_writing_locale_maps_legacy_language_code(base_state: AgentState):
    state = AgentState(**{**base_state, "language": "ja", "writing_locale": "ja"})

    assert resolve_writing_locale(state) == "ja-JP"


def test_zh_planner_context_contains_chinese_style_without_affecting_ja(
    base_state: AgentState,
):
    zh_state = AgentState(**{**base_state, "language": "zh"})
    ja_state = AgentState(**{**base_state, "language": "ja"})

    zh_context = build_planner_style_context(zh_state, "literature_review", "thematic_review")
    ja_context = build_planner_style_context(ja_state, "literature_review", "thematic_review")

    assert "zh-CN" in zh_context
    assert "避免英文句法直译" in zh_context
    assert "ja-JP" in ja_context
    assert "日本語" in ja_context
    assert "避免英文句法直译" not in ja_context


def test_ja_style_pack_includes_paper_type_guidance_and_references(
    base_state: AgentState,
):
    state = AgentState(**{**base_state, "language": "ja"})

    context = build_planner_style_context(state, "literature_review", "thematic_review")

    assert "ja-JP" in context
    assert "研究潮流と未解決課題" in context
    assert "ja_academic_review_1" in context
    assert "文献を列挙するのではなく" in context


def test_style_pack_override_replaces_default_guidance(base_state: AgentState):
    state = AgentState(
        **{
            **base_state,
            "writing_locale": "zh-CN",
            "style_pack_override": {
                "display_name": "Custom zh style",
                "general_guidance": ["Use a school-specific thesis style."],
            },
        }
    )

    context = build_writer_style_context(state)

    assert "Custom zh style" in context
    assert "Use a school-specific thesis style." in context
