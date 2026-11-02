
"""
Qdrant Collection Management
----------------------------
Handles collection creation, existence checks,
collection inspection, and safe deletion.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from vector_store.client import get_qdrant_client

load_dotenv()

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME",
    "hidevs_knowledge_base"
)


def collection_exists(
    collection_name: Optional[str] = None,
    client: Optional[QdrantClient] = None,
) -> bool:
    """
    Check whether a collection exists in Qdrant.

    Args:
        collection_name: Name of the collection.
        client: Optional existing Qdrant client.

    Returns:
        bool: True if the collection exists.
    """

    name = collection_name or COLLECTION_NAME
    qdrant = client or get_qdrant_client()

    try:
        return qdrant.collection_exists(name)

    except Exception:
        logger.exception(
            "Failed to check collection existence: %s", name
        )
        raise


def create_collection(
    vector_size: int,
    collection_name: Optional[str] = None,
    distance: Distance = Distance.COSINE,
    client: Optional[QdrantClient] = None,
) -> bool:
    """
    Create a new Qdrant collection.

    The vector dimension must match the actual output
    dimension of the configured embedding model.

    Args:
        vector_size: Embedding vector dimension.
        collection_name: Collection name.
        distance: Vector distance metric.
        client: Optional existing Qdrant client.

    Returns:
        bool: True if the collection was created.

    Raises:
        ValueError: If the vector dimension is invalid
                    or the collection already exists.
    """

    name = collection_name or COLLECTION_NAME

    if not isinstance(vector_size, int) or isinstance(vector_size, bool):
        raise ValueError("vector_size must be an integer.")

    if vector_size <= 0:
        raise ValueError("vector_size must be greater than zero.")

    qdrant = client or get_qdrant_client()

    if collection_exists(name, qdrant):
        raise ValueError(
            f"Collection '{name}' already exists. "
            "Use get_collection_info() to inspect it."
        )

    try:
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )

        logger.info(
            "Created Qdrant collection '%s' with vector size %d.",
            name,
            vector_size,
        )

        return True

    except Exception:
        logger.exception(
            "Failed to create Qdrant collection: %s", name
        )
        raise


def get_collection_info(
    collection_name: Optional[str] = None,
    client: Optional[QdrantClient] = None,
):
    """
    Retrieve information about an existing collection.

    Returns:
        Collection information from Qdrant.

    Raises:
        ValueError: If the collection does not exist.
    """

    name = collection_name or COLLECTION_NAME
    qdrant = client or get_qdrant_client()

    if not collection_exists(name, qdrant):
        raise ValueError(
            f"Collection '{name}' does not exist."
        )

    try:
        return qdrant.get_collection(name)

    except Exception:
        logger.exception(
            "Failed to retrieve collection information: %s", name
        )
        raise


def delete_collection(
    collection_name: str,
    client: Optional[QdrantClient] = None,
) -> bool:
    """
    Explicitly delete a collection.

    Warning:
        This permanently deletes the collection and its points.
    """

    if not collection_name or not collection_name.strip():
        raise ValueError("A valid collection name is required.")

    qdrant = client or get_qdrant_client()

    if not collection_exists(collection_name, qdrant):
        logger.warning(
            "Collection '%s' does not exist.",
            collection_name,
        )
        return False

    qdrant.delete_collection(collection_name)

    logger.warning(
        "Deleted Qdrant collection: %s",
        collection_name,
    )

    return True
