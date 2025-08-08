import fitz
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.db_mongo.models import Document, Chunk
from app.vector_db.pinecone_client import pinecone_client
from app.core.config import settings

class DocumentService:
    def __init__(self):
        """
        A simplified, robust, and optimized configuration for a high-performance RAG system.
        """
        # Use the most powerful model for generation to maximize accuracy
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
        
        # Use Google's best embedding model
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # Use a balanced chunking strategy
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _get_text_from_source(self, source: str) -> str:
        """Gets text content from either a URL or a local file path."""
        try:
            if source.startswith("http://") or source.startswith("https://"):
                response = requests.get(source, timeout=45) # Increased timeout
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
        """
        Simplified and optimized RAG pipeline.
        """
        document = await Document.find_one(Document.source_url == document_source)
        if not document:
            document = await self._process_new_document(document_source)

        # 1. Embed the user's question
        question_embedding = self.embeddings_model.embed_query(question)

        # 2. Retrieve a larger number of relevant chunks from Pinecone
        pinecone_index = pinecone_client.get_index()
        query_response = pinecone_index.query(
            vector=question_embedding,
            top_k=15, # Retrieve more candidates to ensure context is found
            include_metadata=True
        )
        
        # Add a defensive check for the response structure
        if not query_response or not query_response.get('matches'):
            return "Could not retrieve any information from the document."

        context_chunks = [match['metadata']['text'] for match in query_response['matches']]
        context = "\n---\n".join(context_chunks)
        
        if not context:
            return "Could not find relevant information in the document to answer the question."

        # 3. Use the powerful LLM and an improved prompt to generate the answer
        prompt = f"""
        You are a meticulous and highly accurate assistant specializing in insurance, legal, HR, and compliance.
        Your primary goal is to provide precise, factually grounded answers based only on the provided CONTEXT.
        Follow these steps rigorously:

        1.  **Analyze the QUESTION:** Understand the user's intent and identify the key entities or concepts being asked about.
        2.  **Evaluate CONTEXT Relevance:** Carefully read through each piece of provided CONTEXT. Systematically identify and extract only the sentences or phrases that directly address the QUESTION. Ignore any irrelevant information.
        3.  **Synthesize the Answer:**
            * Combine the extracted, relevant pieces of information to form a comprehensive and coherent answer.
            * *Crucially, do NOT introduce any information not explicitly present in the CONTEXT.*
            * If, after careful evaluation, no sentence in the CONTEXT can answer the QUESTION, you MUST state: "The information is not available in the provided document."
        4.  **Format the FINAL ANSWER:** Present your answer concisely and directly.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        FINAL ANSWER:
        """
        response = await self.llm.ainvoke(prompt)
        return response.content

document_service = DocumentService()
