# app.py

import streamlit as st
import tempfile
import os
from main_pipeline import process_pdf_pipeline

st.set_page_config(page_title="Smart PDF QA", layout="wide")
st.title("🔍 PDF Question Answering System")

uploaded_pdf = st.file_uploader("📄 Upload your PDF", type=["pdf"])

query = st.text_input("🧠 Enter your query", "")

# Strategy selection
strategy = st.selectbox("🧩 Chunking Strategy", ["fixed", "semantic", "custom", "hierarchical"])

# Conditional params for chunking
chunk_size = 512
overlap = 128
semantic_level = "paragraph"

if strategy == "fixed":
    chunk_size = st.slider("Chunk Size", 100, 1000, 512, step=64)
    overlap = st.slider("Overlap", 0, 512, 128, step=32)
elif strategy == "semantic":
    semantic_level = st.selectbox("Semantic Chunk Level", ["paragraph", "sentence"])

# Reranking method
rerank_method = st.selectbox("Reranking Strategy", ["none", "cross-encoder", "mmr"])
top_k = st.slider("Top-K Retrieval", 1, 20, 5)

if st.button("🚀 Run"):
    if uploaded_pdf and query:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(uploaded_pdf.read())
            tmp_pdf_path = tmp_pdf.name

        with st.spinner("Processing..."):
            reranked_docs, answer = process_pdf_pipeline(
                file_path=tmp_pdf_path,
                query=query,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=overlap,
                semantic_level=semantic_level,
                rerank_method=rerank_method,
                top_k=top_k,
                include_embeddings=True
            )

        st.success("✅ Answer generated!")
        st.markdown(f"**📝 Answer:** {answer}")

        st.markdown("---")
        st.subheader("📚 Top Chunks Retrieved and Re-ranked")
        for i, doc in enumerate(reranked_docs):
            st.markdown(f"**Rank {i+1}** (Score: {doc.get('score', 'N/A')})")
            st.markdown(doc["content"])
            st.code(doc["metadata"])
    else:
        st.warning("Please upload a PDF and enter a query.")
