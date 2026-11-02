
import os
import logging

from dotenv import load_dotenv
from langchain_core.documents import Document

from embeddings.gemini_embeddings import GeminiEmbeddingService
from vector_store.client import get_qdrant_client
from vector_store.collections import collection_exists


load_dotenv()

logger = logging.getLogger(__name__)


class VectorSearch:
    """Retrieve relevant knowledge-base chunks from Qdrant."""

    def __init__(self):
        self.collection_name = os.getenv(
            "QDRANT_COLLECTION_NAME",
            "hidevs_knowledge_base",
        )

        self.top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
        self.embedding_dimension = int(
            os.getenv("EMBEDDING_DIMENSION", "3072")
        )

        if self.top_k <= 0:
            raise ValueError("RETRIEVAL_TOP_K must be greater than zero.")

        if self.embedding_dimension <= 0:
            raise ValueError(
                "EMBEDDING_DIMENSION must be greater than zero."
            )

        self.embedding_service = GeminiEmbeddingService()
        self.qdrant_client = get_qdrant_client()

        if not collection_exists(
            collection_name=self.collection_name,
            client=self.qdrant_client,
        ):
            raise RuntimeError(
                f"Qdrant collection '{self.collection_name}' does not exist. "
                "Run the ingestion process first."
            )

    def search(self, query: str) -> list[Document]:
        """Generate a query embedding and retrieve relevant documents."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        query = query.strip()

        # 1. Generate query embedding
        query_embedding = self.embedding_service.embed_query(query)

        # 2. Validate embedding dimension
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dimension}, "
                f"but received {len(query_embedding)}."
            )

        # 3. Retrieve matching points from Qdrant
        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=self.top_k,
            with_payload=True,
        )

        # 4. Convert Qdrant results into LangChain Documents
        documents = []

        for result in response.points:
            payload = result.payload or {}

            page_content = payload.get("page_content")

            if not isinstance(page_content, str) or not page_content.strip():
                logger.warning(
                    "Skipping Qdrant point %s because page_content is missing.",
                    result.id,
                )
                continue

            raw_metadata = payload.get("metadata", {})

            metadata = (
                raw_metadata.copy()
                if isinstance(raw_metadata, dict)
                else {}
            )

            metadata["score"] = float(result.score)
            metadata["point_id"] = str(result.id)

            documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )

        logger.info(
            "Retrieved %d documents for query: %s",
            len(documents),
            query,
        )

        return documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    vector_search = VectorSearch()

    test_questions = [
        "What is CodeQuest and what types of coding problems does it provide?",
        "What is LeetPrompt?",
        "What features are available in Dr. Agent?",
        "What is the purpose of the AI Agents Hub?",
        "What information is available on the HiDevs leaderboard?",
    ]

    for question in test_questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        results = vector_search.search(question)

        for index, document in enumerate(results, start=1):
            print(f"\nRESULT {index}")
            print(f"Score: {document.metadata.get('score')}")
            print(f"Section: {document.metadata.get('section_title')}")
            print(f"Content:\n{document.page_content}")