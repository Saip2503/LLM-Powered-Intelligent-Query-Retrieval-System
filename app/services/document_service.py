import fitz
import requests
import cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.db_mongo.models import Document, Chunk
from app.vector_db.pinecone_client import pinecone_client
from app.core.config import settings

class DocumentService:
    def __init__(self):
        """
        Final optimized configuration for RAG system.
        """
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

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

        # 1. Advanced Query Transformation: Multi-Query Generation
        query_generation_prompt = f"Generate 3 alternative versions of this question for a document search: {question}"
        query_generation_response = await self.llm.ainvoke(query_generation_prompt)
        all_queries = [question] + [q.strip() for q in query_generation_response.content.strip().split('\n') if q.strip()]

        # 2. Hybrid Search: Combine Vector Search (Pinecone) with Keyword Search (BM25)
        bm25_docs = [chunk.text for chunk in document.chunks]
        bm25_retriever = BM25Retriever.from_texts(bm25_docs, k=10)

        retrieved_docs_metadata = {}
        pinecone_index = pinecone_client.get_index()

        # Perform vector search for all queries
        for query_text in all_queries:
            query_embedding = self.embeddings_model.embed_query(query_text)
            query_response = pinecone_index.query(
                vector=query_embedding, top_k=10, include_metadata=True
            )
            
            # --- FIX: Add a check to ensure the response is valid ---
            if query_response and query_response['matches']:
                for match in query_response['matches']:
                    retrieved_docs_metadata[match['id']] = match['metadata']['text']
            # ---------------------------------------------------------

            # Perform BM25 search and add results to the pool
            bm25_results = bm25_retriever.invoke(query_text)
            for doc in bm25_results:
                if doc.page_content not in retrieved_docs_metadata.values():
                    retrieved_docs_metadata[f"bm25_{hash(doc.page_content)}"] = doc.page_content

        initial_docs = list(retrieved_docs_metadata.values())
        if not initial_docs:
            return "Could not find relevant information in the document."

        # 3. Re-rank with Cohere
        context_chunks = []
        try:
            reranked_results = self.cohere_client.rerank(
                query=question, documents=initial_docs, top_n=7, model="rerank-english-v3.0"
            )
            context_chunks = [result.document['text'] for result in reranked_results.results]
        except Exception as e:
            print(f"WARNING: Cohere re-ranker failed: {e}. Falling back to standard retrieval.")
            context_chunks = initial_docs[:7]

        context = "\n---\n".join(context_chunks)

        # 4. Advanced Prompt Engineering for Precision and Factual Integrity
        prompt = f"""
        You are a meticulous and highly accurate assistant specializing in insurance, legal, HR, and compliance.
        Your primary goal is to provide precise, factually grounded answers based only on the provided CONTEXT.
        Follow these steps rigorously:

        1.  *Analyze the QUESTION:* Understand the user's intent and identify key entities or concepts.
        2.  *Evaluate CONTEXT Relevance:* Carefully read through each piece of provided CONTEXT. Determine which sentences or phrases directly address the QUESTION.
        3.  *Synthesize Answer (Step-by-Step - Chain of Thought):*
            * If the QUESTION requires multi-step reasoning, break down your thought process into clear, logical steps.
            * Extract specific sentences or data points from the CONTEXT that directly support each step of your reasoning.
            * Combine these extracted pieces of information to form a comprehensive and coherent answer.
            * *Crucially, do NOT introduce any information not explicitly present in the CONTEXT.*
            * If a direct answer or supporting information is not found in the CONTEXT, state that clearly.
        4.  *Format FINAL ANSWER:* Present your answer concisely and directly.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        FINAL ANSWER:
        """
        response = await self.llm.ainvoke(prompt)
        return response.content

document_service = DocumentService()
