"""Mock HTTP response data for tests."""

ARXIV_SINGLE_ENTRY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2301.00001</id>
    <title>Test Paper Title</title>
    <authors>Author One, Author Two</authors>
    <summary>This is the abstract of the test paper.</summary>
  </entry>
</feed>"""

ARXIV_TWO_ENTRIES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2301.00001</id>
    <title>First Test Paper</title>
    <authors>Author A, Author B</authors>
    <summary>First paper abstract.</summary>
  </entry>
  <entry>
    <id>https://arxiv.org/abs/2301.00002</id>
    <title>Second Test Paper</title>
    <authors>Author C</authors>
    <summary>Second paper abstract.</summary>
  </entry>
</feed>"""

ARXIV_EMPTY_FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""

ARXIV_MISSING_TITLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2301.00001</id>
    <authors>Author One</authors>
    <summary>Abstract without title.</summary>
  </entry>
</feed>"""

OPENALEX_SINGLE_RESULT = {
    "results": [
        {
            "id": "https://openalex.org/W2100001",
            "title": "OpenAlex Test Paper",
            "publication_year": 2022,
            "cited_by_count": 500,
            "authorships": [
                {"author": {"display_name": "Jane Doe"}},
                {"author": {"display_name": "John Smith"}},
            ],
            "abstract_inverted_index": {
                "This": [0],
                "is": [1],
                "the": [2],
                "abstract": [3],
            },
            "doi": "https://doi.org/10.1234/test",
        }
    ]
}

OPENALEX_NULL_ABSTRACT = {
    "results": [
        {
            "id": "https://openalex.org/W2100002",
            "title": "Paper With No Abstract",
            "publication_year": 2021,
            "cited_by_count": 100,
            "authorships": [{"author": {"display_name": "Author X"}}],
            "abstract_inverted_index": None,
            "doi": None,
        }
    ]
}

OPENALEX_EMPTY_RESULTS = {"results": []}

SEMANTIC_SCHOLAR_TWO_PAPERS = {
    "data": [
        {
            "paperId": "s2:paper001",
            "title": "SS Paper One",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "abstract": "SS paper one abstract.",
            "url": "https://semanticscholar.org/paper/001",
            "pdfUrl": None,
            "year": 2023,
            "citationCount": 200,
        },
        {
            "paperId": "s2:paper002",
            "title": "SS Paper Two",
            "authors": [{"name": "Charlie"}],
            "abstract": "SS paper two abstract.",
            "url": "https://semanticscholar.org/paper/002",
            "pdfUrl": None,
            "year": 2020,
            "citationCount": 50,
        },
    ]
}

SEMANTIC_SCHOLAR_EMPTY = {"data": []}
