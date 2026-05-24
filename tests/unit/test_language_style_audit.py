from seele_scholar_agent.nodes.language_style_audit import LanguageStyleAudit
from seele_scholar_agent.state import AgentState


def test_language_style_audit_flags_obvious_zh_translationese(base_state: AgentState):
    state = AgentState(**{**base_state, "language": "zh"})
    findings = LanguageStyleAudit().audit(
        "本文将会讨论该方法如何起到了重要的作用，并说明其有着重要的意义。",
        state,
    )

    codes = {finding.code for finding in findings}
    assert "ZH_TRANSLATIONESE_PHRASE" in codes


def test_language_style_audit_does_not_apply_zh_rules_to_ja(base_state: AgentState):
    state = AgentState(**{**base_state, "language": "ja"})
    findings = LanguageStyleAudit().audit(
        "本文将会讨论该方法如何起到了重要的作用。",
        state,
    )

    assert findings == []


def test_language_style_audit_uses_term_glossary(base_state: AgentState):
    state = AgentState(
        **{
            **base_state,
            "language": "zh",
            "term_glossary": {"大语言模型": "大型语言模型"},
        }
    )

    findings = LanguageStyleAudit().audit("大语言模型的评估需要统一术语。", state)

    assert [finding.code for finding in findings] == ["TERM_GLOSSARY_MISMATCH"]
