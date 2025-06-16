import os
import json
import pickle
from vectorstore.vector_store import META_PATH, INDEX_PATH, embedder, load_index_and_metadata

def export_index_to_json(json_path: str = "output/chunks_summary.json", include_embeddings: bool = False):
    # Ensure output directory exists
    output_dir = os.path.dirname(json_path)
    os.makedirs(output_dir, exist_ok=True)

    # Load stored data
    if not os.path.exists(META_PATH) or not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Required index or metadata file not found.")

    index, chunk_data = load_index_and_metadata()

    result = []

    for i, chunk in enumerate(chunk_data):
        entry = {
            "content": chunk["content"],
            "metadata": chunk["metadata"]
        }

        if include_embeddings:
            vector = index.reconstruct(i).tolist()
            entry["embedding"] = vector[:10]

        result.append(entry)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Exported {len(result)} chunks to {json_path}")
