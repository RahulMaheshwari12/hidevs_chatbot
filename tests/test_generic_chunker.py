
from langchain_core.documents import Document

from ingestion.chunker import KnowledgeBaseChunker


def make_document(text: str) -> Document:
    return Document(
        page_content=text,
        metadata={"source": "synthetic_test.txt"},
    )


def test_numbered_headings():
    text = """
1. Introduction
This is the introduction.

2. Features
Feature information.

2.1 Search
Search feature details.

2.2 Analytics
Analytics feature details.

3. Conclusion
Final information.
"""

    chunker = KnowledgeBaseChunker()
    sections = chunker._parse_sections(make_document(text))

    assert len(sections) == 5
    assert sections[0].metadata["section_number"] == "1"
    assert sections[1].metadata["section_number"] == "2"
    assert sections[2].metadata["section_number"] == "2.1"
    assert sections[2].metadata["parent_section_number"] == "2"
    assert sections[3].metadata["parent_section_number"] == "2"
    assert sections[4].metadata["section_number"] == "3"


def test_lettered_headings():
    text = """
1. Audience
Audience overview.

A. Developers
Developer information.

B. Companies
Company information.

C. Communities
Community information.

2. Conclusion
Final information.
"""

    chunker = KnowledgeBaseChunker()
    sections = chunker._parse_sections(make_document(text))

    lettered = [
        section
        for section in sections
        if section.metadata["section_number"] in ("A", "B", "C")
    ]

    assert len(lettered) == 3

    for section in lettered:
        assert section.metadata["parent_section_number"] == "1"

    assert lettered[0].metadata["parent_section_title"] == "Audience"
    assert lettered[1].metadata["parent_section_title"] == "Audience"


def test_markdown_headings():
    text = """
# Introduction
General introduction.

## Features
Feature details.

### Search
Search details.

## Analytics
Analytics details.

# Conclusion
Final information.
"""

    chunker = KnowledgeBaseChunker()
    sections = chunker._parse_sections(make_document(text))

    assert len(sections) == 5

    search_section = next(
        section
        for section in sections
        if section.metadata["section_title"] == "Search"
    )

    assert search_section.metadata["parent_section_title"] == "Features"


def test_document_without_headings():
    text = """
HiDevs provides learning resources to developers.

Users can access courses and projects.

The platform also provides practical exercises.

Visit https://example.com for more information.

The service launched in August 2024.
"""

    chunker = KnowledgeBaseChunker()
    sections = chunker._parse_sections(make_document(text))

    assert len(sections) == 1
    assert sections[0].metadata["section_title"] == "General Information"
    assert "August 2024" in sections[0].page_content
    assert "https://example.com" in sections[0].page_content


def test_uppercase_heading_and_content():
    text = """
PRODUCT OVERVIEW
This is product information.

The API URL is https://example.com/api.

The launch date is 23 August 2024.

FEATURE DETAILS
These are product features.
"""

    chunker = KnowledgeBaseChunker()
    sections = chunker._parse_sections(make_document(text))

    titles = [section.metadata["section_title"] for section in sections]

    assert "PRODUCT OVERVIEW" in titles
    assert "FEATURE DETAILS" in titles

    all_content = "\n".join(section.page_content for section in sections)

    assert "https://example.com/api" in all_content
    assert "23 August 2024" in all_content


def test_chunking_without_headings():
    text = " ".join(
        f"This is paragraph number {index} containing useful information."
        for index in range(100)
    )

    chunker = KnowledgeBaseChunker()
    chunks = chunker.split_documents([make_document(text)])

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.page_content.strip()
        assert chunk.metadata["section_title"] == "General Information"
        assert chunk.metadata["chunk_index"] >= 1
        assert chunk.metadata["chunk_total"] >= 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(
        pytest.main([__file__, "-v", "-s"])
    )
