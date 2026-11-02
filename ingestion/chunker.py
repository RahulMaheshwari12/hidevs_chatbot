
import os
import re

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


class KnowledgeBaseChunker:
    """Split knowledge-base content into section-aware text chunks."""

    SOURCE_PATTERN = re.compile(
        r"^SOURCE\s+\d+\s+[—-]\s*(.+?)\s*$",
        re.IGNORECASE,
    )

    HEADING_PATTERN = re.compile(
        r"^(\d+(?:\.\d+)*\.?)\s+(.+?)\s*$"
    )

    def __init__(self):
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero.")

        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
            )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            add_start_index=True,
        )

    def _is_section_heading(
        self,
        lines: list[str],
        index: int,
    ) -> bool:
        """Identify standalone numbered section headings."""

        line = lines[index].strip()

        match = self.HEADING_PATTERN.match(line)

        if not match:
            return False

        number = match.group(1)

        # Accept:
        # 1. Platform Master Inventory
        # 15.1 GenAI Internship
        #
        # Reject:
        # 23 Aug 2024: Timeline entry
        # 5 Days Gen AI Workshop

        is_valid_numbering = (
            number.endswith(".") or "." in number
        )

        if not is_valid_numbering:
            return False

        previous_is_blank = (
            index == 0 or not lines[index - 1].strip()
        )

        next_is_blank = (
            index == len(lines) - 1
            or not lines[index + 1].strip()
        )

        return previous_is_blank and next_is_blank

    def _build_sections(
        self,
        document: Document,
    ) -> list[Document]:
        """Identify source and section boundaries."""

        lines = document.page_content.splitlines()

        sections = []

        source_title = "Knowledge Base"
        section_title = "Document Overview"
        section_number = None

        current_lines = []

        def save_section():
            content = "\n".join(current_lines).strip()

            if not content:
                return

            metadata = document.metadata.copy()

            metadata.update({
                "source_title": source_title,
                "section_title": section_title,
                "section_number": section_number,
            })

            sections.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

        for index, line in enumerate(lines):

            # Detect source document boundaries.
            source_match = self.SOURCE_PATTERN.match(line.strip())

            if source_match:
                save_section()

                source_title = source_match.group(1).strip()
                section_title = "Source Overview"
                section_number = None

                current_lines = [line]
                continue

            # Detect valid numbered section headings.
            if self._is_section_heading(lines, index):
                save_section()

                heading_match = self.HEADING_PATTERN.match(
                    line.strip()
                )

                section_number = heading_match.group(1).rstrip(".")
                section_title = heading_match.group(2).strip()

                current_lines = [line]
                continue

            current_lines.append(line)

        save_section()

        if not sections:
            raise ValueError(
                "No sections could be extracted from the knowledge base."
            )

        return sections

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """Split each logical section independently."""

        if not documents:
            raise ValueError("No documents provided for chunking.")

        all_chunks = []

        for document in documents:

            if not document.page_content.strip():
                continue

            sections = self._build_sections(document)

            for section_index, section in enumerate(sections):

                section_chunks = self.splitter.split_documents(
                    [section]
                )

                for chunk_index, chunk in enumerate(section_chunks):

                    chunk.metadata["section_index"] = section_index
                    chunk.metadata["chunk_index"] = chunk_index

                    all_chunks.append(chunk)

        if not all_chunks:
            raise ValueError("No text chunks were generated.")

        return all_chunks


if __name__ == "__main__":
    from ingestion.text_loader import KnowledgeBaseLoader

    loader = KnowledgeBaseLoader()
    chunker = KnowledgeBaseChunker()

    documents = loader.load()
    chunks = chunker.split_documents(documents)

    print("\n========== CHUNKING RESULTS ==========")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks generated: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print("\n" + "=" * 80)
        print(f"CHUNK {index}")
        print(f"Source: {chunk.metadata.get('source_title')}")
        print(f"Section: {chunk.metadata.get('section_title')}")
        print(f"Section Number: {chunk.metadata.get('section_number')}")
        print(f"Chunk Index: {chunk.metadata.get('chunk_index')}")
        print(f"Content Length: {len(chunk.page_content)}")
        print("\nCONTENT:")
        print(chunk.page_content)

    print("\n======================================")
