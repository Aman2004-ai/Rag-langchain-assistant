# app.py

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Ensure the Google API key is set
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment.")

# --- 1. SET UP THE RETRIEVER ---
# Load the previously created vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# --- 2. DEFINE THE PROMPT TEMPLATE ---
template = """
You are a helpful assistant for developers using LangChain.
Answer the question based only on the following context.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- 3. INITIALIZE THE LLM ---
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # <-- Using the latest Flash model
    temperature=0.7, 
    convert_system_message_to_human=True
)
# --- 4. BUILD THE RAG CHAIN ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# --- 5. CREATE A SIMPLE COMMAND-LINE INTERFACE (CLI) ---
if __name__ == "__main__":
    print("Welcome to the LangChain Docs Q&A Assistant!")
    print("Ask a question about LangChain agents, or type 'exit' to quit.")

    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() == "exit":
            print("Goodbye!")
            break
        
        # Invoke the chain with the user's question
        response = rag_chain.invoke(user_question)
        print("\nAssistant:", response)