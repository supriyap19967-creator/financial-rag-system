import streamlit as st
import requests

API_URL = "https://demand-api-912628415543.us-central1.run.app/query"

st.set_page_config(
    page_title="Financial Regulatory RAG",
    layout="wide"
)

st.title("📊 Financial Regulatory Compliance RAG")
st.caption("Hybrid Retrieval (BM25 + FAISS) + CrossEncoder Reranking + Self-RAG Validation")

st.markdown("---")

question = st.text_input("🔎 Ask a compliance-related question")

col1, col2 = st.columns([1, 6])

with col1:
    submit = st.button("Submit")

if submit and question:

    with st.spinner("Retrieving and validating answer..."):
        try:
            response = requests.post(API_URL, json={"question": question}, timeout=60)
            data = response.json()
        except:
            st.error("⚠ Backend API not running. Please start FastAPI server.")
            st.stop()

    st.markdown("---")

    # ✅ Answer Section
    st.subheader("📌 Answer")

    if "insufficient" in data["answer"].lower():
        st.error(data["answer"])
    else:
        st.success(data["answer"])

    st.markdown("---")

    # 📄 Sources Section
    st.subheader("📄 Sources")

    if data["sources"]:
        for source in data["sources"]:
            st.markdown(
                f"- **{source['source']}** | Page: {source['page']} | Chunk: {source['chunk_id']}"
            )
    else:
        st.write("No supporting sources found.")

    st.markdown("---")

    # ⚙ Metadata Section
    st.subheader("⚙ Retrieval Metadata")

    colA, colB = st.columns(2)

    with colA:
        st.metric("Retrieved Chunks", data["retrieved_chunks"])

    with colB:
        st.metric("Latency (seconds)", data["latency_seconds"])

st.markdown("---")
st.caption("Built using FastAPI, FAISS, SentenceTransformers, and Streamlit.")
