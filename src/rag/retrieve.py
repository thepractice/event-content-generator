"""Retrieval functions for the RAG system."""

from __future__ import annotations

import os

import chromadb
from chromadb.utils import embedding_functions


def get_chroma_client() -> chromadb.ClientAPI:
    """Get the ChromaDB client."""
    persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    return chromadb.PersistentClient(path=persist_dir)


def get_embedding_function():
    """Get the default embedding function (uses sentence-transformers locally)."""
    return embedding_functions.DefaultEmbeddingFunction()


def retrieve_chunks(
    query: str,
    collection_name: str = "brand_voice",
    top_k: int = 5,
) -> list[dict]:
    """Retrieve relevant chunks from the vector store.

    Args:
        query: The search query
        collection_name: Which collection to search
        top_k: Number of results to return

    Returns:
        List of chunk dicts with 'id', 'text', 'source' keys
    """
    try:
        client = get_chroma_client()
        embed_fn = get_embedding_function()

        # Get or create collection with embedding function
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Check if collection is empty
        if collection.count() == 0:
            return _get_fallback_chunks(collection_name)

        # Search using query text (ChromaDB will embed it automatically)
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        # Format results
        chunks = []
        for i, doc_id in enumerate(results["ids"][0]):
            chunks.append(
                {
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i].get("source", "unknown"),
                }
            )

        return chunks

    except Exception as e:
        print(f"Warning: Retrieval failed: {e}")
        return _get_fallback_chunks(collection_name)


def _get_fallback_chunks(collection_name: str) -> list[dict]:
    """Return fallback chunks when the vector store is empty or unavailable.

    This ensures the pipeline can still run for testing/demo purposes.
    """
    if collection_name == "brand_voice":
        return [
            {
                "id": "fallback_brand_1",
                "text": """Our brand voice is confident yet approachable. We speak
                directly to our audience as peers, not as authorities lecturing
                from above. Use active voice, concrete examples, and avoid jargon
                unless our audience uses it daily.""",
                "source": "fallback",
            },
            {
                "id": "fallback_brand_2",
                "text": """When writing calls-to-action, be specific about the benefit.
                Don't say 'Learn more' - say 'See how teams cut review time by 50%'.
                Every CTA should answer the reader's question: 'What's in it for me?'""",
                "source": "fallback",
            },
        ]
    elif collection_name == "product_docs":
        return [
            {
                "id": "fallback_product_1",
                "text": """Our platform helps teams collaborate more effectively.
                Key features include real-time editing, version control, and
                seamless integrations with existing workflows.""",
                "source": "fallback",
            },
        ]
    else:
        return []


def search_similar_chunks(
    chunk_id: str,
    collection_name: str = "brand_voice",
    top_k: int = 3,
) -> list[dict]:
    """Find chunks similar to a given chunk.

    Useful for finding additional context around a specific source.

    Args:
        chunk_id: ID of the chunk to find similar content for
        collection_name: Which collection to search
        top_k: Number of results to return

    Returns:
        List of similar chunk dicts
    """
    try:
        client = get_chroma_client()
        embed_fn = get_embedding_function()

        collection = client.get_collection(collection_name, embedding_function=embed_fn)

        # Get the original chunk
        original = collection.get(ids=[chunk_id], include=["embeddings"])

        if not original["embeddings"]:
            return []

        # Search for similar
        results = collection.query(
            query_embeddings=original["embeddings"],
            n_results=top_k + 1,  # +1 because it will include itself
            include=["documents", "metadatas"],
        )

        # Format and filter out the original
        chunks = []
        for i, doc_id in enumerate(results["ids"][0]):
            if doc_id != chunk_id:
                chunks.append(
                    {
                        "id": doc_id,
                        "text": results["documents"][0][i],
                        "source": results["metadatas"][0][i].get("source", "unknown"),
                    }
                )

        return chunks[:top_k]

    except Exception as e:
        print(f"Warning: Similar search failed: {e}")
        return []
