from collections.abc import Awaitable, Callable

from langchain_core.tools import tool


def create_privite_kb_tool(search_func: Callable[[str], Awaitable[str]]):
    @tool
    async def search_private_knowledge(query: str) -> str:
        results = await search_func(query)
        return results

    return search_private_knowledge
