import fitz
import requests
import cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.core.config import settings
from app.db_mongo.model import Document, Chunk
from app.vector_db.pinecone_client import pinecone_client

class DocumentService:
    def __init__(self):
        """
        Final production-ready configuration using the most powerful Gemini APIs.
        """
        # --- Using the most powerful models from Google's API ---
        # Gemini 1.5 Pro for the highest quality reasoning and answer generation
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
        # The latest and best embedding model from Google
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.cohere_client = cohere.Client(settings.COHERE_API_KEY)

    def _get_text_from_source(self, source: str) -> str:
        """Gets text content from either a URL or a local file path."""
        try:
            if source.startswith("http://") or source.startswith("https://"):
                response = requests.get(source, timeout=30)
                response.raise_for_status()
                with fitz.open(stream=response.content, filetype="pdf") as doc:
                    return "".join(page.get_text() for page in doc)
            else:
                with fitz.open(source) as doc:
                    return "".join(page.get_text() for page in doc)
        except Exception as e:
            print(f"Error reading from source {source}: {e}")
            raise

    async def _process_new_document(self, source_url_or_path: str) -> Document:
        """Processes a new document and stores it in the databases."""
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
                "id": chunk.chunk_id, "values": chunk_embeddings[i], "metadata": {"text": chunk.text}
            })

        pinecone_index = pinecone_client.get_index()
        if pinecone_vectors:
            batch_size = 100
            for i in range(0, len(pinecone_vectors), batch_size):
                batch = pinecone_vectors[i:i + batch_size]
                pinecone_index.upsert(vectors=batch)
        
        await new_document.insert()
        print(f"Successfully processed and stored new document with {len(text_chunks)} chunks.")
        return new_document

    async def answer_question(self, document_source: str, question: str) -> str:
        """Final optimized RAG pipeline with Multi-Query, Hybrid Search, and Re-ranking."""
        document = await Document.find_one(Document.source_url == document_source)
        if not document:
            document = await self._process_new_document(document_source)

        # 1. Multi-Query Generation
        query_generation_prompt = f"Generate 3 alternative versions of this question for a document search: {question}"
        query_generation_response = await self.llm.ainvoke(query_generation_prompt)
        all_queries = [question] + [q.strip() for q in query_generation_response.content.strip().split('\n') if q.strip()]

        # 2. Hybrid Search
        all_chunk_texts = [chunk.text for chunk in document.chunks]
        bm25_retriever = BM25Retriever.from_texts(all_chunk_texts, k=10)
        
        retrieved_docs_text = set()
        pinecone_index = pinecone_client.get_index()

        for query_text in all_queries:
            query_embedding = self.embeddings_model.embed_query(query_text)
            query_response = pinecone_index.query(vector=query_embedding, top_k=10, include_metadata=True)
            for match in query_response['matches']:
                retrieved_docs_text.add(match['metadata']['text'])
            
            bm25_results = bm25_retriever.invoke(query_text)
            for doc in bm25_results:
                retrieved_docs_text.add(doc.page_content)

        initial_docs = list(retrieved_docs_text)
        if not initial_docs:
            return "Could not find relevant information in the document."

        # 3. Re-rank with fallback
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
        
        # 4. Final Generation Prompt
        prompt = f"""You are a meticulous assistant. Follow these steps:
        1. Carefully read the QUESTION and the CONTEXT.
        2. Identify the specific sentences from the CONTEXT that directly answer the QUESTION.
        3. If no information is found, your final answer must be "The information is not available in the provided document."
        4. If information is found, synthesize a concise and direct final answer.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        FINAL ANSWER:"""
        response = await self.llm.ainvoke(prompt)
        return response.content

document_service = DocumentService()
