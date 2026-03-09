
from app.ingestion import process_pdf
from app.Vector import create_Vector
import os

file_path = "Data/Finance_RBI.pdf" 
if not os.path.exists(file_path):
    print(f"ERROR: File not found at {file_path}. Please check your Data folder.")

#Run ingestion
chunks = process_pdf(file_path)
print("Total chunks:", len(chunks))

#Validation
print("First chunk preview:\n")
print(chunks[0].page_content[:500])

 

#Check Overlap is working
print("\nLast 150 chars of chunk 1:")
print(chunks[0].page_content[-150:])

print("\nFirst 150 chars of chunk 2:")
print(chunks[1].page_content[:150])

#check metadata give page number
print("\nMetadata of first chunk:")
print(chunks[0].metadata)

#testing: Embeddings, FAISS creation
print("\nCreating vector store...")
Vector= create_Vector(chunks)

print("Vector store created and saved successfully.")




