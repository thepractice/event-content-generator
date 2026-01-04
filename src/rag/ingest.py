"""Corpus ingestion for the RAG system."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


def get_chroma_client() -> chromadb.ClientAPI:
    """Get or create the ChromaDB client with persistence."""
    persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    return chromadb.PersistentClient(path=persist_dir)


def get_embedding_function():
    """Get the default embedding function (uses sentence-transformers locally)."""
    return embedding_functions.DefaultEmbeddingFunction()


def load_corpus(corpus_dir: str = "corpus") -> list[dict]:
    """Load all documents from the corpus directory.

    Args:
        corpus_dir: Path to the corpus directory

    Returns:
        List of document dicts with 'content', 'source', and 'type' keys
    """
    corpus_path = Path(corpus_dir)
    documents = []

    if not corpus_path.exists():
        print(f"Warning: Corpus directory {corpus_dir} does not exist")
        return documents

    # Load all markdown and text files
    for file_path in corpus_path.glob("**/*"):
        if file_path.suffix in [".md", ".txt"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Determine document type from path or filename
            doc_type = _infer_document_type(file_path)

            documents.append(
                {
                    "content": content,
                    "source": str(file_path),
                    "type": doc_type,
                }
            )

    return documents


def _infer_document_type(file_path: Path) -> str:
    """Infer the document type from the file path or name."""
    path_str = str(file_path).lower()

    if "brand" in path_str or "voice" in path_str or "style" in path_str:
        return "brand_voice"
    elif "product" in path_str or "feature" in path_str or "docs" in path_str:
        return "product_docs"
    else:
        return "general"


def chunk_document(
    content: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Split a document into chunks for embedding.

    Args:
        content: Full document content
        source: Source file path
        chunk_size: Target size in characters
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunk dicts with 'id', 'text', 'source' keys
    """
    chunks = []

    # Simple chunking by paragraphs then size
    paragraphs = content.split("\n\n")
    current_chunk = ""
    chunk_index = 0

    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size:
            if current_chunk:
                chunk_id = _generate_chunk_id(source, chunk_index)
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": current_chunk.strip(),
                        "source": source,
                    }
                )
                chunk_index += 1
                # Start new chunk with overlap
                current_chunk = current_chunk[-chunk_overlap:] if chunk_overlap else ""

        current_chunk += para + "\n\n"

    # Don't forget the last chunk
    if current_chunk.strip():
        chunk_id = _generate_chunk_id(source, chunk_index)
        chunks.append(
            {
                "id": chunk_id,
                "text": current_chunk.strip(),
                "source": source,
            }
        )

    return chunks


def _generate_chunk_id(source: str, index: int) -> str:
    """Generate a unique, deterministic chunk ID."""
    content = f"{source}:{index}"
    hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"chunk_{hash_val}"


def ingest_documents(
    corpus_dir: str = "corpus",
    force_reingest: bool = False,
) -> dict:
    """Ingest all documents from corpus into ChromaDB.

    Args:
        corpus_dir: Path to corpus directory
        force_reingest: If True, delete existing collections first

    Returns:
        Dict with ingestion statistics
    """
    client = get_chroma_client()
    embed_fn = get_embedding_function()

    # Load documents
    documents = load_corpus(corpus_dir)

    if not documents:
        return {"status": "warning", "message": "No documents found in corpus"}

    # Separate by type
    brand_docs = [d for d in documents if d["type"] == "brand_voice"]
    product_docs = [d for d in documents if d["type"] == "product_docs"]
    general_docs = [d for d in documents if d["type"] == "general"]

    stats = {"brand_voice": 0, "product_docs": 0, "general": 0}

    # Ingest brand voice documents
    if brand_docs or force_reingest:
        stats["brand_voice"] = _ingest_collection(
            client,
            embed_fn,
            "brand_voice",
            brand_docs + general_docs,  # Include general in brand
            force_reingest,
        )

    # Ingest product documents
    if product_docs or force_reingest:
        stats["product_docs"] = _ingest_collection(
            client,
            embed_fn,
            "product_docs",
            product_docs + general_docs,  # Include general in product
            force_reingest,
        )

    return {"status": "success", "chunks_ingested": stats}


def _ingest_collection(
    client: chromadb.ClientAPI,
    embed_fn,
    collection_name: str,
    documents: list,
    force_reingest: bool,
) -> int:
    """Ingest documents into a specific collection.

    Returns:
        Number of chunks ingested
    """
    if force_reingest:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass  # Collection doesn't exist

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Chunk all documents
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc["content"], doc["source"])
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    # Prepare data for ChromaDB
    texts = [chunk["text"] for chunk in all_chunks]
    ids = [chunk["id"] for chunk in all_chunks]
    metadatas = [{"source": chunk["source"]} for chunk in all_chunks]

    # Add to collection (ChromaDB will generate embeddings automatically)
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
    )

    return len(all_chunks)
