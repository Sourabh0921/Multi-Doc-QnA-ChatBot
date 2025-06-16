from retrieval.evaluation import evaluate_test_set
from vectorstore.vector_store import retrieve_chunks

results = evaluate_test_set(
    test_data_path="evaluation/test_data.json",
    retriever_func=retrieve_chunks,
    k=5,
    similarity_threshold=0.75,
    output_jsonl_path="evaluation/evaluation_results.json"
)

for r in results:
    print("\n--- Evaluation ---")
    print(f"Query: {r['query']}")
    print(f"Precision@5: {r['precision@k']:.2f}")
    print(f"Recall@5: {r['recall@k']:.2f}")
    print(f"F1@5: {r['f1@k']:.2f}")
    print(f"MRR: {r['mrr']:.2f}")
    print(f"NDCG: {r['ndcg']:.2f}")
    print(f"Hit: {r['hit']}")
    print(f"Difficulty: {r['difficulty']}")
