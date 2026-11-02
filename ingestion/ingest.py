
import os
import uuid
import logging

from dotenv import load_dotenv
from qdrant_client.models import PointStruct

from ingestion.text_loader import KnowledgeBaseLoader
from ingestion.chunker import KnowledgeBaseChunker
from embeddings.gemini_embeddings import GeminiEmbeddingService
from vector_store.client import get_qdrant_client
from vector_store.collections import (
    collection_exists,
    create_collection,
)


load_dotenv()

logger = logging.getLogger(__name__)


class KnowledgeBaseIngestion:
    """Load, embed, and safely synchronize the HiDevs knowledge base."""

    def __init__(self):
        self.collection_name = os.getenv(
            "QDRANT_COLLECTION_NAME",
            "hidevs_knowledge_base",
        )

        self.embedding_dimension = int(
            os.getenv("EMBEDDING_DIMENSION", "3072")
        )

        self.loader = KnowledgeBaseLoader()
        self.chunker = KnowledgeBaseChunker()
        self.embedding_service = GeminiEmbeddingService()
        self.qdrant_client = get_qdrant_client()

    def _generate_point_id(self, chunk, index: int) -> str:
        """Generate a deterministic ID for a knowledge-base chunk."""

        source = chunk.metadata.get("source", "unknown")
        source_title = chunk.metadata.get("source_title", "unknown")
        section_number = chunk.metadata.get("section_number", "unknown")
        chunk_index = chunk.metadata.get("chunk_index", index)

        unique_key = (
            f"{source}:{source_title}:"
            f"{section_number}:{chunk_index}:"
            f"{chunk.page_content}"
        )

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                unique_key,
            )
        )

    def _get_existing_point_ids(self) -> set[str]:
        """Retrieve all existing point IDs from the collection."""

        existing_ids = set()
        offset = None

        while True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )

            existing_ids.update(
                str(record.id) for record in records
            )

            if next_offset is None:
                break

            offset = next_offset

        return existing_ids

    def _prepare_collection(self, vector_size: int):
        """Create the collection if it does not already exist."""

        if collection_exists(
            collection_name=self.collection_name,
            client=self.qdrant_client,
        ):
            print("Existing knowledge-base collection found.")
            return

        print("Creating a new knowledge-base collection...")

        create_collection(
            vector_size=vector_size,
            collection_name=self.collection_name,
            client=self.qdrant_client,
        )

    def ingest(self):
        """Ingest and synchronize knowledge-base chunks with Qdrant."""

        # 1. Load knowledge base
        print("\n[1/6] Loading knowledge base...")

        documents = self.loader.load()

        print(f"Documents loaded: {len(documents)}")

        # 2. Generate section-aware chunks
        print("\n[2/6] Generating section-aware chunks...")

        chunks = self.chunker.split_documents(documents)

        if not chunks:
            raise RuntimeError(
                "No chunks were generated. Existing Qdrant data was not changed."
            )

        print(f"Chunks generated: {len(chunks)}")

        # 3. Generate embeddings
        print("\n[3/6] Generating Gemini embeddings...")

        texts = [chunk.page_content for chunk in chunks]

        embeddings = self.embedding_service.embed_documents(texts)

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Embedding count does not match chunk count."
            )

        if not embeddings:
            raise RuntimeError("No embeddings were generated.")

        vector_size = len(embeddings[0])

        if vector_size != self.embedding_dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dimension}, "
                f"but received {vector_size}."
            )

        if any(len(vector) != vector_size for vector in embeddings):
            raise ValueError(
                "Inconsistent embedding dimensions detected."
            )

        print(f"Embedding dimension: {vector_size}")

        # 4. Prepare Qdrant collection
        print("\n[4/6] Preparing Qdrant collection...")

        self._prepare_collection(vector_size)

        # 5. Prepare and upload points
        print("\n[5/6] Uploading vectors and metadata...")

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            point_id = self._generate_point_id(chunk, index)

            payload = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        expected_ids = {
            str(point.id) for point in points
        }

        # Upsert does not delete unrelated existing points.
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        # 6. Verify upload before removing outdated points
        print("\n[6/6] Verifying uploaded vectors...")

        existing_ids = self._get_existing_point_ids()

        missing_ids = expected_ids - existing_ids

        if missing_ids:
            raise RuntimeError(
                f"Upload verification failed. "
                f"{len(missing_ids)} expected points are missing. "
                "Existing points have not been cleaned up."
            )

        # Only remove stale points after all new points are verified.
        stale_ids = existing_ids - expected_ids

        if stale_ids:
            print(
                f"Removing {len(stale_ids)} outdated vectors..."
            )

            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=list(stale_ids),
                wait=True,
            )

        # Final verification
        final_ids = self._get_existing_point_ids()

        if final_ids != expected_ids:
            raise RuntimeError(
                "Final Qdrant verification failed. "
                f"Expected {len(expected_ids)} points, "
                f"found {len(final_ids)}."
            )

        collection_info = self.qdrant_client.get_collection(
            collection_name=self.collection_name
        )

        print("\n========== INGESTION COMPLETE ==========")
        print(f"Collection: {self.collection_name}")
        print(f"Documents loaded: {len(documents)}")
        print(f"Chunks generated: {len(chunks)}")
        print(f"Vectors uploaded: {len(points)}")
        print(f"Outdated vectors removed: {len(stale_ids)}")
        print(f"Verified vectors: {len(final_ids)}")
        print(f"Qdrant points count: {collection_info.points_count}")
        print("=========================================")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ingestion = KnowledgeBaseIngestion()
    ingestion.ingest()
