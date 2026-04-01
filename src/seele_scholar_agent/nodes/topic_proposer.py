from collections.abc import AsyncIterator
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..i18n import t
from ..logging import get_logger
from ..state import AgentState, ProposedTopic
from . import NodeStreamEvent, _stream_llm_text, invoke_with_retry
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
        self.stream_chain = self.prompt | self.llm

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

    async def astream(self, state: AgentState) -> AsyncIterator[NodeStreamEvent]:
        broad_topic = state["topic"]
        lang = state.get("language", "zh")

        yield NodeStreamEvent(type="progress", progress="pre_search")

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

        input_data = {
            "topic": broad_topic,
            "papers_summary": papers_summary,
            "language": t(lang, "language_name"),
        }

        yield NodeStreamEvent(type="progress", progress="generating")

        full_text = ""
        async for event in _stream_llm_text(self.stream_chain, input_data):
            full_text += event.get("token", "")
            yield event

        try:
            parsed = self.parser.parse(full_text)
            proposed_topics = [ProposedTopic(**item) for item in parsed.get("topics", [])]
        except Exception as e:
            logger.error("topic parsing failed", error=str(e))
            proposed_topics = []

        yield NodeStreamEvent(
            type="result",
            result={
                "broad_papers": broad_papers,
                "proposed_topics": proposed_topics,
                "status": "waiting_human",
            },
        )
