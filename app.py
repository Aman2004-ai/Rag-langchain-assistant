# app.py

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Ensure the OpenRouter API key is set
api_key = os.getenv("OPEN_ROUTER_API_KEY")
if not api_key:
    raise ValueError("OPEN_ROUTER_API_KEY not found in .env file or environment.")

# Set up the retriever (with OpenRouter embeddings)
embeddings = OpenAIEmbeddings(
    model="sentence-transformers/all-minilm-l6-v2",
    base_url="https://openrouter.ai/api/v1",  # <-- The change is here
    api_key=api_key,
)

vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Define the prompt template
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

# Initialize the LLM (pointing to OpenRouter)
model = ChatOpenAI(
    model="google/gemini-flash-1.5",
    temperature=0.7,
    base_url="https://openrouter.ai/api/v1",  # <-- The change is here
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "RAG Assistant",
    },
)

# Build the RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Create CLI
if __name__ == "__main__":
    print("Welcome to the LangChain Docs Q&A Assistant (OpenRouter Edition)!")
    print("Ask a question about LangChain agents, or type 'exit' to quit.")

    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() == "exit":
            print("Goodbye!")
            break
        
        response = rag_chain.invoke(user_question)
        print("\nAssistant:", response)
