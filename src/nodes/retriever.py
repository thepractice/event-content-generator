"""Retriever node - pulls relevant brand voice and product context from vector store."""

from __future__ import annotations

from datetime import datetime

from ..schemas import ContentGeneratorState


def retrieve_node(state: ContentGeneratorState) -> ContentGeneratorState:
    """Retrieve relevant chunks from the vector store for RAG grounding.

    This node queries the ChromaDB vector store to find:
    1. Brand voice examples that match the target audience and tone
    2. Product documentation relevant to the event topic

    Args:
        state: Current pipeline state with event details

    Returns:
        Updated state with brand_chunks and product_chunks populated
    """
    from ..rag import retrieve_chunks

    # Build query from event details
    query_text = f"""
    Event: {state['event_title']}
    Description: {state['event_description']}
    Audience: {state['target_audience']}
    Key Messages: {', '.join(state['key_messages'])}
    """

    # Retrieve brand voice examples
    brand_chunks = retrieve_chunks(
        query=query_text,
        collection_name="brand_voice",
        top_k=5,
    )

    # Retrieve product/company information
    product_chunks = retrieve_chunks(
        query=query_text,
        collection_name="product_docs",
        top_k=5,
    )

    # Log the retrieval action
    audit_entry = {
        "node": "retrieve",
        "timestamp": datetime.utcnow().isoformat(),
        "action": "retrieved_chunks",
        "details": {
            "brand_chunks_count": len(brand_chunks),
            "product_chunks_count": len(product_chunks),
            "query_preview": query_text[:200],
        },
    }

    return {
        **state,
        "brand_chunks": brand_chunks,
        "product_chunks": product_chunks,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }
