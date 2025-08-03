import fitz
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.db_mongo.model import Document, Chunk
from app.vector_db.pinecone_client import pinecone_client
import cohere 
from app.core.config import settings
class DocumentService:
    def __init__(self):
        """
        Final optimized configuration.
        """
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        # Use a larger chunk size to keep context together, with a good overlap
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,separators=["\n\n", "\n", ". ", " ", ""])

    # ... (The _get_text_from_source and _process_new_document methods remain the same) ...
    
    def _get_text_from_source(self, source: str) -> str:
        """Gets text content from either a URL or a local file path."""
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
        Final optimized RAG pipeline with Multi-Query, Re-ranking, and Fallback logic.
        """
        document = await Document.find_one(Document.source_url == document_source)
        if not document:
            document = await self._process_new_document(document_source)

        # 1. Multi-Query Generation
        query_generation_prompt = f"Generate 3 alternative versions of this question for a document search: {question}"
        query_generation_response = await self.llm.ainvoke(query_generation_prompt)
        all_queries = [question] + query_generation_response.content.strip().split('\n')
        
        # 2. Retrieve for all queries
        query_embeddings = self.embeddings_model.embed_documents(all_queries)
        
        retrieved_docs_metadata = {}
        pinecone_index = pinecone_client.get_index()
        for embedding in query_embeddings:
            query_response = pinecone_index.query(
                vector=embedding, top_k=10, include_metadata=True
            )
            for match in query_response['matches']:
                retrieved_docs_metadata[match['id']] = match['metadata']['text']
        
        initial_docs = list(retrieved_docs_metadata.values())
        if not initial_docs:
            return "Could not find relevant information in the document."

        # 3. Re-rank with fallback
        context_chunks = []
        try:
            co = cohere.Client(settings.COHERE_API_KEY)
            reranked_results = co.rerank(
                query=question, documents=initial_docs, top_n=7, model="rerank-english-v2.0"
            )
            context_chunks = [result.document['text'] for result in reranked_results.results]
        except Exception as e:
            print(f"WARNING: Cohere re-ranker failed: {e}. Falling back to standard retrieval.")
            context_chunks = initial_docs[:7]

        context = "\n---\n".join(context_chunks)
        
        # 4. Generate with a step-by-step prompt
        prompt = f"""
        You are a meticulous assistant. Follow these steps:
        1. Carefully read the QUESTION and the CONTEXT.
        2. Identify the specific sentences from the CONTEXT that directly answer the QUESTION.
        3. If no information is found, your final answer must be "The information is not available in the provided document."
        4. If information is found, synthesize a concise and direct final answer.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        FINAL ANSWER:
        """
        response = await self.llm.ainvoke(prompt)
        return response.content

document_service = DocumentService()
