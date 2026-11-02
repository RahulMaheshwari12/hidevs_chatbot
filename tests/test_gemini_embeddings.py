
from embeddings.gemini_embeddings import GeminiEmbeddingService


def test_gemini_query_embedding():
    service = GeminiEmbeddingService()

    embedding = service.embed_query(
        "What is HiDevs and what programs does it offer?"
    )

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(value, float) for value in embedding)

    print(f"\nQuery embedding dimension: {len(embedding)}")


def test_gemini_document_embedding():
    service = GeminiEmbeddingService()

    documents = [
        "HiDevs provides AI learning programs.",
        "Students can participate in hackathons."
    ]

    embeddings = service.embed_documents(documents)

    assert len(embeddings) == len(documents)
    assert all(len(vector) > 0 for vector in embeddings)

    dimensions = {len(vector) for vector in embeddings}

    assert len(dimensions) == 1

    print(f"\nDocument embedding dimension: {len(embeddings[0])}")
