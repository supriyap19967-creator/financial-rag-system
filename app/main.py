import logging
import time
import threading
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from app.retriever import load_vector_store
from app.llm import get_hybrid_llm, validate_answer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Financial RAG API")


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = None
reranker = None
llm = None
models_loaded = False


class QueryRequest(BaseModel):
    question: str


# -------- Load models --------
def load_models():
    global vector_store, reranker, llm, models_loaded

    logger.info("Loading models in background...")

    try:
        vector_store = load_vector_store()
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        llm = get_hybrid_llm()

        models_loaded = True
        logger.info("All models loaded successfully.")

    except Exception as e:
        logger.error(f"Error loading models: {e}")


@app.on_event("startup")
def startup_event():
    threading.Thread(target=load_models).start()


# -------- Health check --------
@app.get("/health")
def health():
    return {"status": "ready" if models_loaded else "starting"}


# -------- Root endpoint --------
@app.get("/")
def read_root():
    return {"status": "Financial RAG API running", "models_loaded": models_loaded}


# -------- Query endpoint --------
@app.post("/query")
def query_rag(request: QueryRequest):

    if not models_loaded:
        return {"message": "Models still loading. Try again in a few seconds."}

    start_time = time.time()
    question = request.question

    docs = vector_store.similarity_search(question, k=8)

    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked_docs = [
        doc for _, doc in sorted(zip(scores, docs), reverse=True)
    ]

    context = "\n".join([doc.page_content for doc in ranked_docs[:4]])

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{question}
"""

    answer = llm.invoke(prompt)
    validated_answer = validate_answer(answer)

    latency = round(time.time() - start_time, 2)

    return {
        "question": question,
        "answer": validated_answer,
        "latency_seconds": latency
    }


# -------- Local run --------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port
    )
