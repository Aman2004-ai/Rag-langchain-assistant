import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Ensure the Google API key is set
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment.")

print("Starting the data ingestion process...")

# 1. Load the document
try:
    loader = WebBaseLoader("https://python.langchain.com/v0.1/docs/modules/agents/")
    docs = loader.load()
    print(f"Successfully loaded {len(docs)} document(s).")
except Exception as e:
    print(f"Error loading document: {e}")
    exit()

# 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"Document split into {len(splits)} chunks.")

# 3. Create embeddings and store in FAISS
print("Creating embeddings and building the vector store...")
vectorstore = FAISS.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
print("Vector store created successfully.")

# 4. Save the vector store locally
vectorstore.save_local("faiss_index")
print("Vector store saved locally in the 'faiss_index' folder.")