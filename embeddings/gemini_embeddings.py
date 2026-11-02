
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()


class GeminiEmbeddingService:
    """Generate text embeddings using Google's Gemini embedding model."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            "gemini-embedding-001"
        )

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Check your .env file."
            )

    def _get_embedding_model(self, task_type: str):
        """Initialize the Gemini embedding model for a specific task."""

        return GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=self.api_key,
            task_type=task_type,
        )

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a user query."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Query text cannot be empty.")

        model = self._get_embedding_model("RETRIEVAL_QUERY")
        embedding = model.embed_query(text)

        if not embedding:
            raise RuntimeError("Gemini returned an empty query embedding.")

        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents."""

        if not texts:
            raise ValueError("Document list cannot be empty.")

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every document must contain non-empty text.")

        model = self._get_embedding_model("RETRIEVAL_DOCUMENT")
        embeddings = model.embed_documents(texts)

        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Number of embeddings does not match number of documents."
            )

        return embeddings
