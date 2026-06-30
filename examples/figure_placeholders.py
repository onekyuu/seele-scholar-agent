import re

FIGURE_PATTERN = re.compile(r"\{\{(FIGURE|TABLE): (.+?) \| chunks:\[([^\]]*)\]\}\}")


def extract_figures(content: str) -> list[dict[str, object]]:
    figures: list[dict[str, object]] = []
    for fig_type, description, chunks_str in FIGURE_PATTERN.findall(content):
        chunk_ids = [chunk.strip() for chunk in chunks_str.split(",") if chunk.strip()]
        figures.append(
            {
                "type": fig_type,
                "description": description,
                "chunk_ids": chunk_ids,
            }
        )
    return figures


if __name__ == "__main__":
    sample = (
        "模型性能如下。{{FIGURE: Accuracy comparison across datasets "
        "| chunks:[evidence-openalex-1, evidence-user-2]}}"
    )
    for figure in extract_figures(sample):
        print(figure)
