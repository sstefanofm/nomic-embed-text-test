#!/usr/bin/env python

from pathlib import Path
import numpy as np
import ollama

MODEL_NAME = "nomic-embed-text:v1.5"

def get_embedding(text: str, prefix: str = "search_document") -> list[float]:
    """Generates an embedding vector using Ollama and nomic-embed-text prefixes."""
    response = ollama.embed(
        model=MODEL_NAME,
        input=f"{prefix}: {text}",
    )
    return response.embeddings[0]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculates cosine similarity between two vectors (1.0 = identical meaning)."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def index_documents(docs_dir: Path) -> list[dict]:
    """Reads all .txt files in a directory and generates embeddings for them."""
    indexed_docs = []
    txt_files = list(docs_dir.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {docs_dir.resolve()}")
        return []

    print(f"Indexing {len(txt_files)} document(s)...")

    for file_path in txt_files:
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        # Generate embedding with 'search_document' prefix for storage
        embedding = get_embedding(content, prefix="search_document")

        indexed_docs.append(
            {
                "file_name": file_path.name,
                "content": content,
                "embedding": embedding,
            }
        )

    print("Indexing complete!\n")
    return indexed_docs


def search(query: str, indexed_docs: list[dict], top_k: int = 3):
    """Searches indexed documents using a query string."""
    if not indexed_docs:
        print("No documents indexed.")
        return

    # Generate query embedding using 'search_query' prefix
    query_vector = get_embedding(query, prefix="search_query")

    # Calculate similarity for each document
    results = []
    for doc in indexed_docs:
        score = cosine_similarity(query_vector, doc["embedding"])
        results.append((score, doc))

    # Sort results by similarity score in descending order
    results.sort(key=lambda x: x[0], reverse=True)

    print(f"--- Search Results for: '{query}' ---")
    for score, doc in results[:top_k]:
        print(f"Score: {(score * 100):.2f}% | File: {doc['file_name']}")
        print(f"Snippet: {doc['content'][:120]}...\n")


if __name__ == "__main__":
    # 1. Create a dummy 'data' directory with sample files if it doesn't exist
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    (data_dir / "physics.txt").write_text(
        "The blue color of the atmosphere is due to Rayleigh scattering of light by particles."
    )
    (data_dir / "cooking.txt").write_text(
        "To make a delicious chocolate cake, combine cocoa powder, flour, sugar, and baking powder."
    )
    (data_dir / "space.txt").write_text(
        "Astronomers use optical and radio telescopes to observe distant galaxies across the universe."
    )

    # 2. Index local documents
    docs = index_documents(data_dir)

    # 3. Perform semantic searches
    search("Why is the sky blue?", docs)
    search("How do I bake a sweet dessert?", docs)
