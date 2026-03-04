import time
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from app.retriever import load_vector_store
from sentence_transformers import CrossEncoder
from app.llm import get_hybrid_llm, validate_answer

# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI()

# Load Retriever & Reranker once at startup
retriever = load_vector_store()
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------------------
# Health Check Endpoint
# ---------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "financial-regulatory-rag",
        "model_ready": True
    }


# ---------------------------
# Request Schema
# ---------------------------
class QueryRequest(BaseModel):
    question: str


# ---------------------------
# Main Query Endpoint
# ---------------------------
@app.post("/query")
def query_rag(request: QueryRequest):
    start_time = time.time()

    logger.info(f"Question received: {request.question}")

    # 1️⃣ Hybrid Retrieval
    docs = retriever.invoke(request.question)

    if not docs:
        logger.warning("No documents retrieved.")
        return {
            "answer": "I don't have enough relevant information in the document.",
            "sources": [],
            "retrieved_chunks": 0,
            "latency_seconds": 0
        }

    logger.info(f"Initial retrieved chunks: {len(docs)}")

    # 2️⃣ Reranking
    pairs = [(request.question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    # Log rerank scores
    for doc, score in scored_docs:
        logger.info(f"Rerank score: {round(float(score),3)} | Page: {doc.metadata.get('page')}")

    # 3️⃣ Threshold Filtering
    THRESHOLD = 0.25
    filtered_docs = [doc for doc, score in scored_docs if score >= THRESHOLD]

    if not filtered_docs:
        logger.warning("No chunk passed threshold. Using top 1 fallback.")
        filtered_docs = [scored_docs[0][0]]

    docs = filtered_docs[:3]

    logger.info(f"Chunks after filtering: {len(docs)}")

    # 4️⃣ Build Context
    context = "\n\n".join([doc.page_content for doc in docs])

    # 5️⃣ Extract Citation Metadata
    sources = []
    for i, doc in enumerate(docs):
        sources.append({
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "chunk_id": i + 1
        })

    # 6️⃣ Strict Generation Prompt
    prompt = f"""
You are a regulatory compliance analyst.

Answer ONLY using the provided context.

You MUST:
- Explicitly mention the exact Paragraph number if present.
- Explicitly mention the exact Rule number if present.
- Explicitly mention the reporting authority (e.g., FIU-IND).
- Do NOT summarize away clause numbers.

If paragraph or rule number is found in context,
you MUST include it in the answer.

If no specific paragraph or rule is found,
say: "Specific paragraph not available in retrieved context."

Context:
{context}

Question:
{request.question}
"""

    # 7️⃣ LLM Generation
    answer = get_hybrid_llm(prompt)

    # 8️⃣ Self-RAG Validation
    validation = validate_answer(context, answer)
    logger.info(f"Validation result: {validation}")

    if "YES" not in validation.upper():
        logger.warning("Validation failed. Returning safe response.")
        answer = "The retrieved context is insufficient to provide a reliable answer."

    end_time = time.time()
    latency = round(end_time - start_time, 2)

    logger.info(f"Latency: {latency} seconds")

    # 9️⃣ Final Response
    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(docs),
        "latency_seconds": latency
    }