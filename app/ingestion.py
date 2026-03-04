from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def process_pdf(file_path: str):
    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Chunking strategy
    #We use a smaller overlap but ensure we keep headers together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
	add_start_index=True  # tracks position in doc
    )

    chunks = text_splitter.split_documents(documents)

# Metadata

    file_name = os.path.basename(file_path)
    for chunk in chunks:
        chunk.metadata["source"]=file_name
    return chunks
	      
if __name__ == "__main__":
    path = r"C:\Users\supri\Desktop\Financial Rag API\Data\Finance_RBI.pdf"
    print("Starting ingestion...")
    result = process_pdf(path)
    print(f"Success! Created {len(result)} chunks.") 
  