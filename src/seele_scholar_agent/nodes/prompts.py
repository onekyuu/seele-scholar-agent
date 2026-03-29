LANGUAGE_NAMES = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
}

LANGUAGE_TITLES = {
    "zh": "论文标题",
    "en": "Paper Title",
    "ja": "論文タイトル",
}

LANGUAGE_ABSTRACT = {
    "zh": "摘要",
    "en": "Abstract",
    "ja": "要約",
}

LANGUAGE_KEYWORDS = {
    "zh": "关键词",
    "en": "Keywords",
    "ja": "キーワード",
}

PLANNER_SYSTEM_PROMPT = """你是一位资深的学术论文大纲规划师。根据给定的研究主题和检索到的相关文献，生成一份{language}的{language_title}大纲。

要求：
1. 遵循标准学术论文结构 (Introduction -> Related Work -> Method -> Experiment -> Conclusion)
2. 章节数量适中 (6-10个主要章节)，每个章节有明确的描述
3. 适当引用检索到的文献
4. 输出有效的 JSON 格式"""

PLANNER_USER_PROMPT = """研究主题：{topic}

检索到的相关文献：
{papers_summary}

请生成{language}的论文大纲，使用{language_title}：
{{
    "title": "{title_placeholder}",
    "abstract": "{abstract_placeholder}",
    "sections": [
        {{
            "title": "章节标题",
            "description": "章节描述",
            "order": 1,
            "key_points": ["关键论点1", "关键论点2"]
        }}
    ],
    "keywords": ["{keyword_placeholder}1", "{keyword_placeholder}2", "{keyword_placeholder}3"]
}}

只输出 JSON，不要有其他文字"""

WRITER_SYSTEM_PROMPT = """你是一位专业的学术论文撰写者。你擅长撰写严谨、清晰、论证充分的学术论文内容。

要求：
1. 学术风格，严谨客观
2. 适当引用相关工作，使用 [1], [2] 格式
3. 使用 Markdown 格式输出，但不要输出章节标题（章节标题由系统自动添加）
4. 直接输出内容正文，不要有任何解释性文字
5. 使用{language}撰写
6. 不要在内容中包含 # 或 ## 标题符号"""

WRITER_USER_PROMPT = """论文主题：{topic}

目标语言：{language}

当前章节：{section_title}
章节描述：{section_description}

论文大纲：
{outline_json}

相关文献上下文：
{rag_context}

历史审稿意见：
{review_comments}

请使用{language}撰写该章节的完整内容。不要输出章节标题，只输出正文内容。"""

REVIEWER_SYSTEM_PROMPT = """
你是一位资深的学术论文审稿人。你会严格审查论文内容，给出建设性的修改意见。
输出有效的 JSON 格式。"""

REVIEWER_USER_PROMPT = """请审阅以下论文章节：

论文主题：{topic}
章节标题：{section_title}
章节内容：
{content}

请给出 JSON 格式的审阅结果：
{{
    "approved": true或false,
    "score": 1-10,
    "issues": [
        {{
            "type": "factual_error|missing_citation|weak_argument|format_issue",
            "description": "问题描述",
            "suggestion": "修改建议"
        }}
    ],
    "summary": "总体审阅意见"
}}

如果 score >= 7 且 issues 为空，approved 应为 true。"""
