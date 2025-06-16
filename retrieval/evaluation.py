from sentence_transformers import SentenceTransformer, util
import json
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def is_text_similar(a: str, b: str):
    emb_a = model.encode(a, convert_to_tensor=True)
    emb_b = model.encode(b, convert_to_tensor=True)
    score = util.cos_sim(emb_a, emb_b).item()
    return score

def evaluate_test_set(test_data_path: str, retriever_func, k: int = 5, similarity_threshold: float = 0.75, output_jsonl_path: str = None):
    with open(test_data_path, 'r') as f:
        test_cases = json.load(f)

    # Clear the JSONL file if already exists
    if output_jsonl_path:
        open(output_jsonl_path, 'w', encoding='utf-8').close()

    results = []

    for case in test_cases:
        query = case["query"]
        relevant_texts = case["relevant_texts"]
        print(f"\n==============================")
        print(f"🔍 Query: {query}")
        print(f"📌 Ground Truth Texts:")
        for i, gt in enumerate(relevant_texts):
            print(f"  GT-{i+1}: {gt}")

        retrieved_docs = retriever_func(query, top_k=k)
        retrieved_texts = [doc["content"] for doc in retrieved_docs]

        is_relevant = []
        detailed_retrieved_chunks = []

        for i, text in enumerate(retrieved_texts):
            print(f"\n🔹 Retrieved [{i+1}]: {text[:150].strip()}...")

            chunk_detail = {
                "content": text,
                "score_with_gt": [],
                "is_relevant": False
            }

            for gt_text in relevant_texts:
                score = is_text_similar(text, gt_text)
                matched = score >= similarity_threshold

                chunk_detail["score_with_gt"].append({
                    "gt": gt_text,
                    "cosine_score": round(score, 4),
                    "match": matched
                })

                if matched:
                    print(f"✅ Match with: \"{gt_text[:100].strip()}...\" (Score: {score:.2f})")
                    chunk_detail["is_relevant"] = True
                else:
                    print(f"❌ Not matched with: \"{gt_text[:100].strip()}\" (Score: {score:.2f})")

            is_relevant.append(chunk_detail["is_relevant"])
            detailed_retrieved_chunks.append(chunk_detail)

        # Calculate metrics
        precision = sum(is_relevant[:k]) / k
        recall = sum(is_relevant[:k]) / len(relevant_texts)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        mrr = next((1 / (i + 1) for i, flag in enumerate(is_relevant[:k]) if flag), 0.0)
        dcg = sum([1 / np.log2(i + 2) if rel else 0 for i, rel in enumerate(is_relevant[:k])])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_texts), k))])
        ndcg = dcg / idcg if idcg > 0 else 0.0
        hit = int(any(is_relevant[:k]))

        # print(f"\n📊 Scores:")
        # print(f"Precision@{k}: {precision:.2f}")
        # print(f"Recall@{k}: {recall:.2f}")
        # print(f"F1@{k}: {f1:.2f}")
        # print(f"MRR: {mrr:.2f}")
        # print(f"NDCG: {ndcg:.2f}")
        # print(f"Hit: {hit}")
        # print(f"Difficulty: {case.get('difficulty', 'unknown')}")

        result = {
            "query": query,
            "ground_truths": relevant_texts,
            "retrieved_chunks": detailed_retrieved_chunks,
            "precision@k": precision,
            "recall@k": recall,
            "f1@k": f1,
            "mrr": mrr,
            "ndcg": ndcg,
            "hit": hit,
            "difficulty": case.get("difficulty", "unknown")
        }

        results.append(result)

        if output_jsonl_path:
            with open(output_jsonl_path, 'w', encoding='utf-8') as out_f:
                json.dump(results, out_f, indent=2, ensure_ascii=False)

    return results
