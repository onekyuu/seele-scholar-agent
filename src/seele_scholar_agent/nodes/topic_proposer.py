from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..i18n import t
from ..logging import get_logger
from ..state import AgentState, ProposedTopic
from . import invoke_with_retry
from .researcher import SemanticScholarRetriever

logger = get_logger(__name__)


class TopicProposerNode:
    def __init__(
        self, llm: ChatOpenAI, prompts: PromptsConfig, senmatic_scholar_key: str | None = None
    ):
        self.llm = llm
        self.prompts = prompts
        self.retriever = SemanticScholarRetriever(api_key=senmatic_scholar_key, top_k=5)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts.topic_proposer_system_prompt),
                ("user", self.prompts.topic_proposer_user_prompt),
            ]
        )
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser

    async def propose(self, state: AgentState) -> dict[str, Any]:
        broad_topic = state["topic"]
        lang = state.get("language", "zh")
        logger.info("exploring broad topic with pre-search", topic=broad_topic)

        try:
            broad_papers = await self.retriever.search(broad_topic)
        except Exception as e:
            logger.error("pre-search failed", error=str(e))
            broad_papers = []

        if broad_papers:
            papers_summary = "\n\n".join(
                [f"- **{p.title}** ({p.authors[0]}等): {p.abstract[:200]}..." for p in broad_papers]
            )
        else:
            papers_summary = t(lang, "no_recent_papers")

        logger.info("generating specific topics from research trends")

        try:
            result = await invoke_with_retry(
                self.chain,
                {
                    "topic": broad_topic,
                    "papers_summary": papers_summary,
                    "language": t(lang, "language_name"),
                },
            )

            proposed_topics = [ProposedTopic(**item) for item in result.get("topics", [])]
        except Exception as e:
            logger.error("topic generation failed after retries", error=str(e))
            proposed_topics = []

        return {
            "broad_papers": broad_papers,
            "proposed_topics": proposed_topics,
            "status": "waiting_human",
        }
