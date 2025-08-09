import fitz
import requests
import cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.db_mongo.models import Document, Chunk
from app.vector_db.pinecone_client import pinecone_client
from app.core.config import settings

class DocumentService:
    def __init__(self):
        """Final, high-accuracy RAG configuration with a robust two-stage retrieval pipeline."""
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
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
        """Processes a new document with standard chunking."""
        print(f"Processing new document from {source_url_or_path}...")
        document_text = self._get_text_from_source(source_url_or_path)
        
        if not document_text:
             print(f"WARNING: No text could be extracted from {source_url_or_path}. Skipping.")
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
                "id": chunk.chunk_id,
                "values": chunk_embeddings[i],
                "metadata": {"text": chunk.text}
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
        """High-accuracy RAG pipeline with a two-stage retrieval (Vector Search + Re-ranker)."""
        document = await Document.find_one(Document.source_url == document_source)
        if not document:
            document = await self._process_new_document(document_source)

        if not document.chunks:
            return "Error: The document is empty or contains no readable text. Cannot process the query."

        # 1. Broad Vector Search
        pinecone_index = pinecone_client.get_index()
        query_embedding = self.embeddings_model.embed_query(question)
        query_response = pinecone_index.query(vector=query_embedding, top_k=20, include_metadata=True)
        
        if not query_response or not query_response.get('matches'):
            return "Could not find relevant information in the document."

        initial_docs = [match['metadata']['text'] for match in query_response['matches']]
        
        # 2. Precise Re-ranking
        reranked_results = self.cohere_client.rerank(
            query=question, documents=initial_docs, top_n=7, model="rerank-english-v3.0"
        )
        
        context_chunks = [
            result.document['text'] for result in reranked_results.results if result.document
        ]
        context = "\n---\n".join(context_chunks)
        
        # 3. Generate Final Answer
        prompt = f"""
        You are a meticulous assistant. Your task is to answer the user's question based *exclusively* on the provided context.
        Analyze the context step-by-step to find the most relevant information.
        If the answer is not present, you must state: "The information is not available in the provided document."
        Provide a direct and concise final answer.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        FINAL ANSWER:
        """
        response = await self.llm.ainvoke(prompt)
        return response.content

document_service = DocumentService()
