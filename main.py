# main.py

from fastapi import FastAPI, Depends, HTTPException
from auth import verify_token # Import the dependency
from pydantic import BaseModel
from typing import List
import fitz  # PyMuPDF
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

app = FastAPI()

# Define your Pydantic models
class SubmissionRequest(BaseModel):
    documents: str
    questions: List[str]

class SubmissionResponse(BaseModel):
    answers: List[str]

@app.post(
    "/hackrx/run",
    response_model=SubmissionResponse,
    dependencies=[Depends(verify_token)] 
)
async def run_submission(request: SubmissionRequest):
    """
    This function now contains the full RAG pipeline logic.
    """
    # 1. Load and parse the document
    document_text = load_document_from_url(request.documents)
    if not document_text:
        # Handle cases where document loading fails
        raise HTTPException(status_code=400, detail="Could not process the document from the URL.")

    # 2. Create the RAG chain
    rag_chain = create_rag_chain(document_text)
    
    # 3. Process all questions and collect answers
    answers = []
    for question in request.questions:
        try:
            response = await rag_chain.ainvoke({"input": question})
            answers.append(response.get("answer", "Error: Could not generate an answer."))
        except Exception as e:
            answers.append(f"An error occurred while processing this question: {e}")
            
    # 4. Return the final response
    return SubmissionResponse(answers=answers)


def load_document_from_url(url: str) -> str:
    """Downloads a PDF from a URL and extracts its text."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        # In a real app, you'd raise an HTTPException here
        print(f"Error loading document: {e}")
        return ""

def create_rag_chain(document_text: str):
    """Creates a RAG chain using Google Gemini models."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = text_splitter.split_text(document_text)
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based ONLY on the provided context.
        If the answer is not found in the context, state that the information is not available.
        
        CONTEXT:
        {context}

        QUESTION:
        {input}
        """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)