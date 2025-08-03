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
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

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
        """Answers a question using an ADVANCED RAG pipeline with Multi-Query retrieval."""
        
        # Ensure the document is processed and in our databases
        document = await Document.find_one(Document.source_url == document_source)
        if not document:
            document = await self._process_new_document(document_source)

        # 1. Set up the base retriever from Pinecone
        pinecone_index = pinecone_client.get_index()
        vector_store = ... # This part requires a LangChain Pinecone wrapper. 
                           # Let's simplify for this example by performing the multi-query manually.

        # --- MANUAL MULTI-QUERY IMPLEMENTATION ---

        # 1a. Use an LLM to generate alternative questions
        query_generation_prompt = f"""
        You are an AI language model assistant. Your task is to generate 3 different versions of the given user
        question to retrieve relevant documents from a vector database. By generating multiple perspectives
        on the user question, your goal is to help the user overcome some of the limitations of distance-based
        similarity search. Provide these alternative questions separated by newlines.

        Original question: {question}
        """
        query_generation_response = await self.llm.ainvoke(query_generation_prompt)
        all_queries = [question] + query_generation_response.content.strip().split('\n')
        
        print(f"Generated Queries: {all_queries}")

        # 1b. Embed all questions and retrieve documents for each
        query_embeddings = self.embeddings_model.embed_documents(all_queries)
        
        retrieved_docs_metadata = {}
        for embedding in query_embeddings:
            query_response = pinecone_index.query(
                vector=embedding,
                top_k=7, # Retrieve 7 docs for each query
                include_metadata=True
            )
            for match in query_response['matches']:
                # Use the chunk_id (match['id']) to avoid duplicates
                retrieved_docs_metadata[match['id']] = match['metadata']['text']
        
        initial_docs = list(retrieved_docs_metadata.values())
        
        if not initial_docs:
            return "Could not find relevant information in the document to answer the question."

        # 2. Re-rank the combined, unique results using Cohere
        co = cohere.Client(settings.COHERE_API_KEY)
        print(f"Re-ranking {len(initial_docs)} combined documents...")
        reranked_results = co.rerank(
            query=question,
            documents=initial_docs,
            top_n=5,  # Return the top 5 most relevant documents
            model="rerank-english-v2.0"
        )

        # 3. Build the final context from the top re-ranked documents
        context_chunks = [result.document['text'] for result in reranked_results.results]
        context = "\n---\n".join(context_chunks)
        
        # 4. Generate the final answer with Gemini
        prompt = f"""
        You are a meticulous assistant. Answer the question based ONLY on the provided context.
        If the answer is not found, state: "The information is not available in the provided document."

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        response = await self.llm.ainvoke(prompt)
        return response.content
document_service = DocumentService()
