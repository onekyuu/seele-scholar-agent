import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from seele_scholar_agent.config import settings
from seele_scholar_agent.graph import create_simple_writing_graph, create_writing_graph
from seele_scholar_agent.state import AgentState

OUTPUT_DIR = Path(__file__).parent / "output"

LANGUAGE_NAMES = {"zh": "中文", "en": "英文", "ja": "日文"}


def save_paper(result: dict, output_dir: Path = OUTPUT_DIR, lang: str = "zh") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    outline = result.get("outline")
    sections = result.get("sections", [])

    if not outline:
        raise ValueError("No outline in result")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lang_tag = LANGUAGE_NAMES.get(lang, "中文")
    filename = f"[{lang_tag}]{outline.title[:30].replace(' ', '_')}_{timestamp}.md"
    filepath = output_dir / filename

    lines = [
        f"# {outline.title}",
        "",
        f"**Keywords:** {', '.join(outline.keywords)}",
        "",
        "## Abstract",
        "",
        outline.abstract,
        "",
    ]

    if outline.sections:
        lines.append("## Outline")
        for s in sorted(outline.sections, key=lambda x: x.order):
            lines.append(f"- **{s.title}**: {s.description}")
        lines.append("")

    if sections:
        lines.append("---")
        lines.append("")
        lines.append("# Paper Content")
        lines.append("")

        for section in sorted(sections, key=lambda x: x.order_index):
            lines.append(f"## {section.title}")
            lines.append("")
            if section.content:
                lines.append(section.content)
            else:
                lines.append("*Content pending*")
            lines.append("")
            lines.append("---")
            lines.append("")

    filepath.write_text("\n".join(lines))
    return filepath


def build_initial_state(topic: str, lang: str, thread_id: str) -> AgentState:
    return {
        "thread_id": thread_id,
        "topic": topic,
        "language": lang,
        "created_at": datetime.now(),
        "tenant_id": None,
        "papers": [],
        "search_queries": [],
        "outline": None,
        "outline_approved": True,
        "sections": [],
        "current_section_index": 0,
        "sections_completed": [],
        "review_history": [],
        "current_review": None,
        "rag_context": [],
        "status": "idle",
        "error_message": None,
        "max_revisions": settings.MAX_REVISIONS,
        "revision_count": 0,
    }


async def run_with_human_approval(topic: str, lang: str):
    print("=" * 60)
    print("模式一：人工确认大纲后继续")
    print("=" * 60)

    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None,
        base_url=settings.OPENAI_BASE_URL,
        temperature=settings.OPENAI_TEMPERATURE,
    )
    app = create_writing_graph(model=model)

    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = build_initial_state(topic, lang, thread_id)
    initial_state["outline_approved"] = False

    print("\n[Step 1] 运行 graph直到 waiting_human...")
    result = await app.ainvoke(initial_state, config=config)
    print(f"状态: {result.get('status')}")

    if result.get("outline"):
        print("\n生成的大纲:")
        print(f"  标题: {result['outline'].title}")
        print(f"  章节数: {len(result['outline'].sections)}")
        for s in result["outline"].sections:
            print(f"    {s.order}. {s.title}")

    if result.get("status") == "waiting_human":
        print("\n[Step 2] 用户确认大纲，更新状态...")
        app.update_state(config, {"outline_approved": True})

        print("[Step 3] 继续执行剩余流程...")
        result = await app.ainvoke(None, config=config)
        print(f"最终状态: {result.get('status')}")

        if result.get("sections"):
            print(f"\n撰写的章节 ({len(result['sections'])}):")
            for s in result["sections"]:
                print(f"  - {s.title}: {s.status}")


async def run_auto_approved(topic: str, lang: str):
    print("\n" + "=" * 60)
    print(f"自动确认大纲 - 主题: {topic} - 语言: {LANGUAGE_NAMES.get(lang, lang)}")
    print("=" * 60)

    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None,
        base_url=settings.OPENAI_BASE_URL,
        temperature=settings.OPENAI_TEMPERATURE,
    )
    app = create_simple_writing_graph(model=model, openalex_email="ai@keyu.email")

    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = build_initial_state(topic, lang, thread_id)

    print("\n运行完整流程...")
    result = await app.ainvoke(initial_state, config=config)
    print(f"状态: {result.get('status')}")

    if result.get("outline"):
        print(f"标题: {result['outline'].title}")
        print(f"章节数: {len(result['outline'].sections)}")

    if result.get("sections"):
        print("章节状态:")
        for s in result["sections"]:
            content_preview = s.content[:50] + "..." if len(s.content) > 50 else s.content
            print(f"  {s.title}: {s.status}")

    print("\n保存论文到文件...")
    output_path = save_paper(result, lang=lang)
    print(f"已保存: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Seele Scholar Agent")
    parser.add_argument("--topic", "-t", default="生态警务现实困境与发展面向", help="研究主题")
    parser.add_argument(
        "--lang",
        "-l",
        default="zh",
        choices=["zh", "en", "ja"],
        help="论文语言: zh=中文, en=英文, ja=日文",
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="auto",
        choices=["auto", "human"],
        help="运行模式: auto=自动, human=人工确认",
    )
    args = parser.parse_args()

    print("seele-scholar-agent 演示")
    print(f"主题: {args.topic}")
    print(f"语言: {LANGUAGE_NAMES.get(args.lang, args.lang)}")
    print()

    if args.mode == "human":
        await run_with_human_approval(args.topic, args.lang)
    else:
        await run_auto_approved(args.topic, args.lang)


if __name__ == "__main__":
    asyncio.run(main())
