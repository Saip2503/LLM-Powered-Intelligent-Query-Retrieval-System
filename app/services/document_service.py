import fitz
import requests
import torch
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever
# --- UPDATED IMPORTS ---
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# -----------------------
from app.db_mongo.models import Document, Chunk
from app.vector_db.pinecone_client import pinecone_client
import cohere
from app.core.config import settings

class DocumentService:
    def __init__(self):
        """
        Final optimized configuration for RAG system using a self-hosted GPT model.
        """
        # --- MODEL CONFIGURATION ---
        generator_model_id = "openai/gpt-oss-20b"
        embedding_model_id = "sentence-transformers/all-mpnet-base-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading models to device: {device}")

        # --- 1. LOAD GENERATOR MODEL (GPT-OSS-20B) ---
        # This requires a powerful GPU and significant VRAM.
        generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_id)
        generator_model = AutoModelForCausalLM.from_pretrained(
            generator_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto", # Automatically uses available GPUs
        )
        pipe = pipeline(
            "text-generation",
            model=generator_model,
            tokenizer=generator_tokenizer,
            max_new_tokens=512,
            temperature=0.1,
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

        # --- 2. LOAD EMBEDDING MODEL ---
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_model_id,
            model_kwargs={'device': device}
        )
        
        # --- 3. OTHER COMPONENTS ---
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.cohere_client = cohere.Client(settings.COHERE_API_KEY)

    def _get_text_from_source(self, source: str) -> str:
        """
        Gets text content from either a URL or a local file path.
        """
        try:
            if source.startswith("http://") or source.startswith("https://"):
                print(f"Downloading from URL: {source}")
                response = requests.get(source, timeout=30)
                response.raise_for_status()
                with fitz.open(stream=response.content, filetype="pdf") as doc:
                    return "".join(page.get_text() for page in doc)
            else:
                print(f"Reading from local file: {source}")
                with fitz.open(source) as doc:
                    return "".join(page.get_text() for page in doc)
        except Exception as e:
            print(f"Error reading from source {source}: {e}")
            raise

    async def _process_new_document(self, source_url_or_path: str) -> Document:
        """
        Processes a new document and stores it in the databases.
        """
        print(f"Processing new document from {source_url_or_path}...")
        document_text = self._get_text_from_source(source_url_or_path)

        text_chunks = self.text_splitter.split_text(document_text)
        document_chunks = [Chunk(text=t) for t in text_chunks]
        new_document = Document(source_url=source_url_or_path, chunks=document_chunks)

        chunk_texts_for_embedding = [chunk.text for chunk in document_chunks]
        chunk_embeddings = self.embeddings_model.embed_documents(chunk_texts_for_embedding)

        pinecone_vectors = []
        for i, chunk in enumerate(document_chunks):
            pinecone_vectors.append({
                "id": chunk.chunk_id,
                "values": chunk_embeddings[i],
                "metadata": {"text": chunk.text}
            })

        pinecone_index = pinecone_client.get_index()
        if pinecone_vectors:
            batch_size = 100
            print(f"Uploading {len(pinecone_vectors)} vectors in batches of {batch_size}...")
            for i in range(0, len(pinecone_vectors), batch_size):
                batch = pinecone_vectors[i:i + batch_size]
                print(f"Upserting batch {i//batch_size + 1}...")
                pinecone_index.upsert(vectors=batch)

        await new_document.insert()

        print(f"Successfully processed and stored new document with {len(text_chunks)} chunks.")
        return new_document

    async def answer_question(self, document_source: str, question: str) -> str:
        """
        Final optimized RAG pipeline with Multi-Query, Hybrid Search, Re-ranking, and Advanced Prompting.
        """
        document = await Document.find_one(Document.source_url == document_source)
        if not document:
            document = await self._process_new_document(document_source)

        query_generation_prompt = f"Generate 3 alternative versions of this question for a document search: {question}"
        
        loop = asyncio.get_running_loop()
        query_generation_response = await loop.run_in_executor(None, lambda: self.llm.invoke(query_generation_prompt))
        
        all_queries = [question] + [q.strip() for q in query_generation_response.strip().split('\n') if q.strip()]

        all_chunk_texts = [chunk.text for chunk in document.chunks]
        bm25_retriever = BM25Retriever.from_texts(all_chunk_texts, k=10)

        retrieved_docs_text = set()
        pinecone_index = pinecone_client.get_index()

        for query_text in all_queries:
            query_embedding = self.embeddings_model.embed_query(query_text)
            query_response = pinecone_index.query(
                vector=query_embedding, top_k=10, include_metadata=True
            )
            for match in query_response['matches']:
                retrieved_docs_text.add(match['metadata']['text'])

            bm25_results = bm25_retriever.invoke(query_text)
            for doc in bm25_results:
                retrieved_docs_text.add(doc.page_content)

        initial_docs = list(retrieved_docs_text)
        if not initial_docs:
            return "Could not find relevant information in the document."

        context_chunks = []
        try:
            print(f"Re-ranking {len(initial_docs)} combined documents...")
            reranked_results = self.cohere_client.rerank(
                query=question, documents=initial_docs, top_n=7, model="rerank-english-v2.0"
            )
            context_chunks = [result.document['text'] for result in reranked_results.results]
        except Exception as e:
            print(f"WARNING: Cohere re-ranker failed: {e}. Falling back to standard retrieval.")
            context_chunks = initial_docs[:7]

        context = "\n---\n".join(context_chunks)

        prompt = f"""
        You are a meticulous and highly accurate assistant...
        CONTEXT:
        {context}
        QUESTION:
        {question}
        FINAL ANSWER:
        """
        
        # Run the final generation in a thread to avoid blocking
        response = await loop.run_in_executor(None, lambda: self.llm.invoke(prompt))
        return response

document_service = DocumentService()
