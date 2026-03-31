from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..logging import get_logger
from ..state import AgentState, ProposedTopic
from .researcher import SemanticScholarRetriever

logger = get_logger(__name__)


class TopicProposerNode:
    def __init__(
        self, model: ChatOpenAI, prompts: PromptsConfig, senmatic_scholar_key: str | None = None
    ):
        self.model = model
        self.prompts = prompts
        self.retriever = SemanticScholarRetriever(api_key=senmatic_scholar_key, top_k=5)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts.topic_proposer_system_prompt),
                ("user", self.prompts.topic_proposer_user_prompt),
            ]
        )
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.model | self.parser

    async def propose(self, state: AgentState) -> dict[str, Any]:
        broad_topic = state["topic"]
        lang = state.get("language", "zh")
        logger.info(f"正在为宽泛主题进行前置检索探索: {broad_topic}")

        try:
            broad_papers = await self.retriever.search(broad_topic)

        except Exception as e:
            logger.error(f"前置检索失败：{e}")
            broad_papers = []

        if broad_papers:
            papers_summary = "\n\n".join(
                [f"- **{p.title}** ({p.authors[0]}等): {p.abstract[:200]}..." for p in broad_papers]
            )
        else:
            papers_summary = "未检索到最新文献，请直接基于常识进行推演。"

        logger.info("正在基于研究趋势生成具体选题...")

        try:
            result = await self.chain.ainvoke(
                {
                    "topic": broad_topic,
                    "papers_summary": papers_summary,
                    "language": self.prompts.language_names.get(lang, "中文"),
                }
            )

            proposed_topics = [ProposedTopic(**t) for t in result.get("topics", [])]
        except Exception as e:
            logger.error(f"选题生成失败: {e}")
            proposed_topics = []

        return {
            "broad_papers": broad_papers,
            "proposed_topics": proposed_topics,
            "status": "waiting_human",
        }
