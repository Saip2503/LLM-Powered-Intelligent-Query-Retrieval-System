import fitz
import requests
import cohere
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from app.db_mongo.models import Document, Chunk
from app.vector_db.pinecone_client import pinecone_client
from app.core.config import settings

class DocumentService:
    def __init__(self):
        """Final, robust RAG configuration with a specialized embedding model."""
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': device}
        )

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.cohere_client = cohere.Client(settings.COHERE_API_KEY)

    def _get_text_from_source(self, source: str) -> str:
        """Gets text content from a URL or local file, robust against errors."""
        try:
            if source.startswith("http://") or source.startswith("https://"):
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(source, headers=headers, timeout=45)
                response.raise_for_status()
                with fitz.open(stream=response.content, filetype="pdf") as doc:
                    return "".join(page.get_text() for page in doc if page.get_text())
            else:
                with fitz.open(source) as doc:
                    return "".join(page.get_text() for page in doc if page.get_text())
        except Exception as e:
            print(f"Error reading from source {source}: {e}")
            raise

    async def _process_new_document(self, source_url_or_path: str) -> Document:
        """Processes a new document with the new embedding model."""
        print(f"Processing new document from {source_url_or_path}...")
        document_text = self._get_text_from_source(source_url_or_path)
        
        if not document_text:
             print(f"WARNING: No text could be extracted. Skipping.")
             empty_doc = Document(source_url=source_url_or_path, chunks=[])
             await empty_doc.insert()
             return empty_doc

        text_chunks = self.text_splitter.split_text(document_text)
        document_chunks = [Chunk(text=t) for t in text_chunks]
        new_document = Document(source_url=source_url_or_path, chunks=document_chunks)
        
        chunk_texts_for_embedding = [chunk.text for chunk in document_chunks]
        chunk_embeddings = self.embeddings_model.embed_documents(chunk_texts_for_embedding)

        pinecone_vectors = []
        for i, chunk in enumerate(document_chunks):
            pinecone_vectors.append({
                "id": chunk.chunk_id, "values": chunk_embeddings[i], "metadata": {"text": chunk.text}
            })

        pinecone_index = pinecone_client.get_index()
        if pinecone_vectors:
            batch_size = 100
            for i in range(0, len(pinecone_vectors), batch_size):
                pinecone_index.upsert(vectors=pinecone_vectors[i:i + batch_size])
        
        await new_document.insert()
        print(f"Successfully processed document with {len(text_chunks)} chunks.")
        return new_document

    async def answer_question(self, document_source: str, question: str) -> str:
        """High-accuracy RAG pipeline with a robust embedding model and re-ranking."""
        document = await Document.find_one(Document.source_url == document_source)
        if not document:
            document = await self._process_new_document(document_source)

        if not document.chunks:
            return "Error: The document is empty or contains no readable text."

        # 1. Vector Search
        pinecone_index = pinecone_client.get_index()
        query_embedding = self.embeddings_model.embed_query(question)
        query_response = pinecone_index.query(vector=query_embedding, top_k=20, include_metadata=True)
        
        # --- THIS IS THE FIX ---
        # Use attribute access (.matches) instead of dictionary access (['matches'])
        if not query_response or not query_response.matches:
            return "Could not find relevant information in the document."

        # Use attribute access (.metadata) for each match
        initial_docs = [
            match.metadata['text'] 
            for match in query_response.matches 
            if match.metadata and 'text' in match.metadata
        ]
        # --- END OF FIX ---
        
        # 2. Re-rank the results
        reranked_results = self.cohere_client.rerank(
            query=question, documents=initial_docs, top_n=7, model="rerank-english-v3.0"
        )
        
        context_chunks = [
            result.document['text'] for result in reranked_results.results if result.document
        ]
        context = "\n---\n".join(context_chunks)
        
        # 3. Generate Final Answer
        prompt = f"""
        You are a meticulous assistant. Analyze the CONTEXT to answer the QUESTION.
        If the answer is not present, state: "The information is not available in the provided document."

        CONTEXT:
        {context}

        QUESTION:
        {question}

        FINAL ANSWER:
        """
        response = await self.llm.ainvoke(prompt)
        return response.content

document_service = DocumentService()
