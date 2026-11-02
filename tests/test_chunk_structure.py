
from ingestion.text_loader import KnowledgeBaseLoader
from ingestion.chunker import KnowledgeBaseChunker


def test_print_chunk_structure():
    loader = KnowledgeBaseLoader()
    chunker = KnowledgeBaseChunker()

    documents = loader.load()
    chunks = chunker.split_documents(documents)

    print(f"\nTotal chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata

        print("\n" + "=" * 70)
        print(f"CHUNK: {index}")

        print(f"Source: {metadata.get('source')}")
        print(f"Source document: {metadata.get('source_document')}")
        print(
            f"Source document number: "
            f"{metadata.get('source_document_number')}"
        )

        print(f"Section number: {metadata.get('section_number')}")
        print(f"Section title: {metadata.get('section_title')}")

        print(
            f"Parent section number: "
            f"{metadata.get('parent_section_number')}"
        )
        print(
            f"Parent section title: "
            f"{metadata.get('parent_section_title')}"
        )

        print(f"Topic: {metadata.get('topic')}")

        print(
            f"Chunk position: "
            f"{metadata.get('chunk_index')}/"
            f"{metadata.get('chunk_total')}"
        )

        print(f"Start index: {metadata.get('start_index')}")
        print(f"Length: {len(chunk.page_content)}")

        print("\nContent preview:")
        print(chunk.page_content[:250])


if __name__ == "__main__":
    test_print_chunk_structure()
