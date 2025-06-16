# main_pipeline.py

from chunking.chunker import chunk_pdf_document
from retrieval.evaluation import evaluate_test_set
from vectorstore.vector_store import embed_and_store_chunks, retrieve_chunks
from reranker.rerank import rerank
from llm_model.llm_integration import truncate_context, generate_answer
from utils.export_visualization import export_index_to_json

def process_pdf_pipeline(file_path, query, strategy="semantic", chunk_size=512, overlap=128, semantic_level="paragraph", rerank_method="mmr", top_k=5, include_embeddings=True):
    # Step 1: Chunking
    chunks = chunk_pdf_document(
        file_path=file_path,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        semantic_level=semantic_level
    )

    # Step 2: Embedding & Storage
    embed_and_store_chunks(chunks)

    

    # Step 3: Retrieval
    retrieved_docs = retrieve_chunks(query, top_k=top_k)

    # Step 4: Reranking
    final_docs = rerank(query, retrieved_docs, top_k=top_k, method=rerank_method)

    # Step 5: LLM Answer
    context_text = truncate_context(final_docs, max_tokens=1500)
    final_answer = generate_answer(query, context_text)

    # Step 6: Export for inspection (optional)
    if include_embeddings:
        export_index_to_json("output/chunks_with_embeddings.json", include_embeddings=True)

    return final_docs, final_answer
