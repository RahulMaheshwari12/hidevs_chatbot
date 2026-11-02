
"""
Qdrant Client Connection
------------------------
Manages the connection between the HiDevs Support Chatbot
and the Qdrant vector database.
"""

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """
    Create and return a cached Qdrant client instance.

    Returns:
        QdrantClient: Connected Qdrant client.

    Raises:
        ValueError: If required configuration is missing.
        ConnectionError: If Qdrant is unreachable.
    """

    api_key = os.getenv("QDRANT_API_KEY") or None

    # A full URL (e.g. a hosted Qdrant Cloud endpoint over HTTPS) takes
    # precedence. Otherwise fall back to a local host and port.
    url = os.getenv("QDRANT_URL", "").strip()

    if url:
        if not url.startswith(("http://", "https://")):
            raise ValueError("QDRANT_URL must start with http:// or https://.")
    else:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = os.getenv("QDRANT_PORT", "6333")

        if not host:
            raise ValueError("QDRANT_HOST is not configured.")

        try:
            port = int(port)
        except ValueError as exc:
            raise ValueError("QDRANT_PORT must be a valid integer.") from exc

        url = f"http://{host}:{port}"

    try:
        client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=10,
        )

        # Verify that the Qdrant server is reachable.
        client.get_collections()

        logger.info("Successfully connected to Qdrant at %s", url)

        return client

    except Exception as exc:
        logger.exception("Failed to connect to Qdrant at %s", url)
        raise ConnectionError(
            f"Could not connect to Qdrant at {url}. "
            "Check whether the Docker container is running "
            "and the port mapping is correct."
        ) from exc


def check_qdrant_connection() -> bool:
    """
    Check whether the Qdrant server is accessible.

    Returns:
        bool: True if the connection is successful.
    """

    try:
        client = get_qdrant_client()
        client.get_collections()

        logger.info("Qdrant connection health check passed.")
        return True

    except Exception:
        logger.exception("Qdrant connection health check failed.")
        return False
