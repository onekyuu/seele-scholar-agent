from typing import Any

_STRINGS: dict[str, dict[str, Any]] = {
    "zh": {
        "language_name": "中文",
        "language_title": "论文标题",
        "language_abstract": "摘要",
        "language_keywords": "关键词",
        "no_papers_found": "无相关文献",
        "no_recent_papers": "未检索到最新文献，请直接基于常识进行推演。",
        "review_round": "【第 {round} 轮审稿】评分：{score}/10",
        "review_opinion": "意见: {summary}",
        "review_issue": "问题 {i}: [{type}] {description}",
        "review_suggestion": "建议: {suggestion}",
        "review_error_summary": "审稿过程发生错误",
        "review_error_retry": "请重试",
        "default_paper_title": "关于 {topic} 的研究",
        "default_sections": ["引言", "相关工作", "方法", "实验", "结论"],
        "default_section_descs": ["研究背景", "文献综述", "提出方法", "实验结果", "总结"],
        "references_header": "参考文献",
        "finalizer_abstract_instruction": "请根据已完成的论文各章节内容，撰写一段精炼的摘要。",
        "finalizer_conclusion_instruction": "请根据已完成的论文各章节内容，撰写结论章节。",
        "consistency_no_issues": "各章节一致性良好，未发现明显问题。",
        "citation_alignment_issue": "引用内容与论文不匹配",
    },
    "en": {
        "language_name": "English",
        "language_title": "Paper Title",
        "language_abstract": "Abstract",
        "language_keywords": "Keywords",
        "no_papers_found": "No relevant papers found",
        "no_recent_papers": "No recent papers found. Please reason from general knowledge.",
        "review_round": "[Round {round}] Score: {score}/10",
        "review_opinion": "Opinion: {summary}",
        "review_issue": "Issue {i}: [{type}] {description}",
        "review_suggestion": "Suggestion: {suggestion}",
        "review_error_summary": "An error occurred during review",
        "review_error_retry": "Please retry",
        "default_paper_title": "Research on {topic}",
        "default_sections": [
            "Introduction",
            "Related Work",
            "Methodology",
            "Experiment",
            "Conclusion",
        ],
        "default_section_descs": [
            "Background",
            "Literature Review",
            "Proposed Method",
            "Experimental Results",
            "Conclusion",
        ],
        "references_header": "References",
        "finalizer_abstract_instruction": (
            "Based on the completed sections, write a concise abstract."
        ),
        "finalizer_conclusion_instruction": (
            "Based on the completed sections, write the conclusion."
        ),
        "consistency_no_issues": "No consistency issues detected across sections.",
        "citation_alignment_issue": "Citation content does not match the referenced paper",
    },
    "ja": {
        "language_name": "日本語",
        "language_title": "論文タイトル",
        "language_abstract": "要旨",
        "language_keywords": "キーワード",
        "no_papers_found": "関連文献なし",
        "no_recent_papers": "最新の文献が見つかりませんでした。一般知識から推論してください。",
        "review_round": "【第 {round} 回査読】スコア：{score}/10",
        "review_opinion": "意見: {summary}",
        "review_issue": "問題 {i}: [{type}] {description}",
        "review_suggestion": "提案: {suggestion}",
        "review_error_summary": "査読中にエラーが発生しました",
        "review_error_retry": "再試行してください",
        "default_paper_title": "{topic}に関する研究",
        "default_sections": ["序論", "関連研究", "方法", "実験", "結論"],
        "default_section_descs": ["研究背景", "文献レビュー", "提案手法", "実験結果", "結論"],
        "references_header": "参考文献",
        "finalizer_abstract_instruction": (
            "完成した各セクションの内容に基づいて、簡潔な要旨を書いてください。"
        ),
        "finalizer_conclusion_instruction": (
            "完成した各セクションの内容に基づいて、結論を書いてください。"
        ),
        "consistency_no_issues": "セクション間の一貫性に問題は見つかりませんでした。",
        "citation_alignment_issue": "引用内容が参照論文と一致しません",
    },
}

_FALLBACK = "zh"


def t(lang: str, key: str, **kwargs: Any) -> str:
    table = _STRINGS.get(lang) or _STRINGS[_FALLBACK]
    text: Any = table.get(key) or _STRINGS[_FALLBACK].get(key, key)
    if not isinstance(text, str):
        return key
    return text.format(**kwargs) if kwargs else text


def t_list(lang: str, key: str) -> list[str]:
    table = _STRINGS.get(lang) or _STRINGS[_FALLBACK]
    value: Any = table.get(key) or _STRINGS[_FALLBACK].get(key, [])
    return value if isinstance(value, list) else []
