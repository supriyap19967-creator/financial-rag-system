import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_Vector(chunks, persist_path="data/Vector"):
    # Ensure directory
    if not os.path.exists("data"):
        os.makedirs("data")
    
    #  professional choice 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Creating Vector Store
    vector_db = FAISS.from_documents(chunks, embeddings)

    # Save locally to avoid re-paying for embeddings every time
    vector_db.save_local(persist_path)

    return vector_db

if __name__ == "__main__":
    from ingestion import process_pdf
    
    # 1. Get the chunks first
    print("Step 1: Getting chunks from PDF...")
    path = r"C:\Users\supri\Desktop\Financial Rag API\Data\Finance_RBI.pdf"
    chunks = process_pdf(path)
    
    # 2. Run the vector creation
    print("Step 2: Starting Vector Store creation...")
    create_Vector(chunks)
    
    print("Step 3: Verification - Check the 'data/Vector' folder now!")