import os
import faiss
import pickle
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

# Select your embedding model here
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "vectorstore/faiss_index.bin"
META_PATH = "vectorstore/chunk_metadata.pkl"

# Load model globally to avoid reloading every time
embedder = SentenceTransformer(EMBED_MODEL_NAME)


def embed_and_store_chunks(chunks: List[Dict], persist: bool = True) -> None:
    texts = [chunk["content"] for chunk in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # ⬇️ Save full chunk info (with content + metadata)
    full_chunks = [{"content": chunk["content"], "metadata": chunk["metadata"]} for chunk in chunks]

    if persist:
        faiss.write_index(index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(full_chunks, f)

    print(f"✅ Stored {len(texts)} chunks into FAISS index.")



def load_index_and_metadata() -> Tuple[faiss.Index, List[Dict]]:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Vector index or metadata file not found.")

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def retrieve_chunks(query: str, top_k: int = 5) -> List[Dict]:
    index, all_chunks = load_index_and_metadata()

    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    retrieved = []
    for idx in indices[0]:
        if idx < len(all_chunks):
            retrieved.append(all_chunks[idx])
    return retrieved

