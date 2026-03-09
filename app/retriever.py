from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

def load_vector_store(persist_path="Data/Vector"):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        persist_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # FAISS retriever
    faiss_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 30
        }
    )

    # BM25 retriever
    docs = vector_store.similarity_search("test", k=100)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 6

    # Hybrid retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )

    return hybrid_retriever
