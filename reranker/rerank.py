from typing import List, Dict, Literal
from sentence_transformers import CrossEncoder, util
import numpy as np
import torch

# Load cross-encoder model (for re-ranking)
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def cross_encoder_rerank(query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
    """Re-rank based on cross-encoder scores."""
    pairs = [[query, doc["content"]] for doc in documents]
    scores = cross_encoder_model.predict(pairs)
    
    # Attach scores and sort
    for doc, score in zip(documents, scores):
        doc["score"] = float(score)
    ranked = sorted(documents, key=lambda x: x["score"], reverse=True)
    
    return ranked[:top_k]


def mmr_rerank(
    query: str, 
    documents: List[Dict], 
    top_k: int = 5, 
    diversity_lambda: float = 0.5, 
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> List[Dict]:
    from sentence_transformers import SentenceTransformer, util
    import torch

    embed_model = SentenceTransformer(embed_model_name)
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    doc_embeddings = embed_model.encode([doc["content"] for doc in documents], convert_to_tensor=True)

    selected = []
    selected_idx = []
    remaining_idx = list(range(len(documents)))
    similarity = util.cos_sim(query_embedding, doc_embeddings)[0]

    for _ in range(min(top_k, len(documents))):
        if len(selected) == 0:
            idx = int(torch.argmax(similarity))
        else:
            selected_embeds = doc_embeddings[selected_idx]
            diversity = util.cos_sim(doc_embeddings, selected_embeds).max(dim=1).values
            mmr = (1 - diversity_lambda) * similarity - diversity_lambda * diversity
            mmr[selected_idx] = -1e9  # to avoid reselecting
            idx = int(torch.argmax(mmr))

        selected.append(documents[idx])
        selected_idx.append(idx)
        documents[idx]["score"] = float(similarity[idx])  # 👈 Add this line to store relevance score

    return selected



def rerank(
    query: str,
    documents: List[Dict],
    top_k: int = 5,
    method: Literal["cross-encoder", "mmr"] = "cross-encoder"
) -> List[Dict]:
    """Main strategy switcher"""
    if method == "cross-encoder":
        return cross_encoder_rerank(query, documents, top_k)
    elif method == "mmr":
        return mmr_rerank(query, documents, top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")
