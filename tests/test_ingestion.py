
from ingestion.text_loader import KnowledgeBaseLoader
from ingestion.chunker import KnowledgeBaseChunker


def test_knowledge_base_loading():
    loader = KnowledgeBaseLoader()

    documents = loader.load()

    assert len(documents) > 0
    assert all(document.page_content.strip() for document in documents)

    print(f"\nLoaded documents: {len(documents)}")
    print(f"Source metadata: {documents[0].metadata}")


def test_knowledge_base_chunking():
    loader = KnowledgeBaseLoader()
    chunker = KnowledgeBaseChunker()

    documents = loader.load()
    chunks = chunker.split_documents(documents)

    assert len(chunks) > 0
    assert all(chunk.page_content.strip() for chunk in chunks)

    print(f"\nTotal chunks generated: {len(chunks)}")
    print(f"First chunk length: {len(chunks[0].page_content)}")
    print(f"First chunk preview:\n{chunks[0].page_content[:300]}")


def test_chunk_metadata_preservation():
    loader = KnowledgeBaseLoader()
    chunker = KnowledgeBaseChunker()

    documents = loader.load()
    chunks = chunker.split_documents(documents)

    source = documents[0].metadata.get("source")

    assert all(chunk.metadata.get("source") == source for chunk in chunks)
