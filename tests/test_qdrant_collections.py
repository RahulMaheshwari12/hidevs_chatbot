
import pytest

from qdrant_client.models import Distance

from vector_store.client import get_qdrant_client
from vector_store.collections import (
    collection_exists,
    create_collection,
    get_collection_info,
    delete_collection,
)


TEST_COLLECTION = "hidevs_test_collection"


@pytest.fixture
def qdrant_client():
    return get_qdrant_client()


@pytest.fixture
def cleanup_test_collection(qdrant_client):
    # Remove leftover test collection before testing.
    if collection_exists(TEST_COLLECTION, qdrant_client):
        delete_collection(TEST_COLLECTION, qdrant_client)

    yield

    # Clean up after testing.
    if collection_exists(TEST_COLLECTION, qdrant_client):
        delete_collection(TEST_COLLECTION, qdrant_client)


def test_collection_creation(qdrant_client, cleanup_test_collection):
    result = create_collection(
        vector_size=768,
        collection_name=TEST_COLLECTION,
        distance=Distance.COSINE,
        client=qdrant_client,
    )

    assert result is True
    assert collection_exists(TEST_COLLECTION, qdrant_client)


def test_collection_info(qdrant_client, cleanup_test_collection):
    create_collection(
        vector_size=768,
        collection_name=TEST_COLLECTION,
        client=qdrant_client,
    )

    info = get_collection_info(TEST_COLLECTION, qdrant_client)

    assert info.config.params.vectors.size == 768
    assert info.config.params.vectors.distance == Distance.COSINE


def test_duplicate_collection_creation(
    qdrant_client,
    cleanup_test_collection,
):
    create_collection(
        vector_size=768,
        collection_name=TEST_COLLECTION,
        client=qdrant_client,
    )

    with pytest.raises(ValueError):
        create_collection(
            vector_size=768,
            collection_name=TEST_COLLECTION,
            client=qdrant_client,
        )


def test_collection_deletion(qdrant_client, cleanup_test_collection):
    create_collection(
        vector_size=768,
        collection_name=TEST_COLLECTION,
        client=qdrant_client,
    )

    result = delete_collection(TEST_COLLECTION, qdrant_client)

    assert result is True
    assert not collection_exists(TEST_COLLECTION, qdrant_client)

