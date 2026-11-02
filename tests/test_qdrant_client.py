from vector_store.client import (
    get_qdrant_client,
    check_qdrant_connection,
)

if __name__ == "__main__":
    if check_qdrant_connection():
        client = get_qdrant_client()

        print("\nQdrant connected successfully!")

        collections = client.get_collections()

        print("Available collections:")
        for collection in collections.collections:
            print(f"- {collection.name}")

    else:
        print("Qdrant connection failed.")