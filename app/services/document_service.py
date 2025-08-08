import fitz
import requests
import cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.db_mongo.models import Document, ParentChunk, ChildChunk
from app.vector_db.pinecone_client import pinecone_client
from app.core.config import settings

class DocumentService:
    def __init__(self):
        """High-accuracy configuration using Parent Document chunking and Hybrid Search."""
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # Splitter for large, context-rich parent chunks
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        # Splitter for small, precise child chunks for vector search
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        
        self.cohere_client = cohere.Client(settings.COHERE_API_KEY)

    def _get_text_from_source(self, source: str) -> str:
                """
                Gets text content from either a URL or a local file path,
                now with a User-Agent header to mimic a browser.
                """
                try:
                    if source.startswith("http://") or source.startswith("https://"):
                        print(f"Downloading from URL: {source}")
                        
                        # --- FIX: Add a User-Agent header ---
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }
                        response = requests.get(source, headers=headers, timeout=45)
                        # ------------------------------------
                        
                        response.raise_for_status()
                        
                        # Check if the content is actually a PDF
                        if 'application/pdf' not in response.headers.get('Content-Type', ''):
                            print(f"WARNING: URL did not return a PDF. Content-Type: {response.headers.get('Content-Type')}")
                            # You might want to handle this more gracefully, but for now, we'll let fitz try
                        
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
        """Processes a new document using the Parent Document chunking strategy."""
        print(f"Processing new document with Parent-Child strategy from {source_url_or_path}...")
        document_text = self._get_text_from_source(source_url_or_path)
        
        # 1. Create large parent chunks
        parent_texts = self.parent_splitter.split_text(document_text)
        
        new_document = Document(source_url=source_url_or_path, parent_chunks=[])
        
        all_child_chunks = []
        pinecone_vectors = []

        # 2. Create small child chunks for each parent
        for parent_text in parent_texts:
            parent_chunk = ParentChunk(text=parent_text, children=[])
            
            child_texts = self.child_splitter.split_text(parent_text)
            for child_text in child_texts:
                child_chunk = ChildChunk(text=child_text)
                parent_chunk.children.append(child_chunk)
                all_child_chunks.append(child_chunk)

            new_document.parent_chunks.append(parent_chunk)

        # 3. Embed ONLY the child chunks
        child_texts_for_embedding = [chunk.text for chunk in all_child_chunks]
        child_embeddings = self.embeddings_model.embed_documents(child_texts_for_embedding)

        for i, child_chunk in enumerate(all_child_chunks):
            pinecone_vectors.append({
                "id": child_chunk.chunk_id,
                "values": child_embeddings[i],
                "metadata": {"text": child_chunk.text} # Store child text for BM25
            })

        # 4. Batch upsert child vectors to Pinecone
        pinecone_index = pinecone_client.get_index()
        if pinecone_vectors:
            batch_size = 100
            for i in range(0, len(pinecone_vectors), batch_size):
                pinecone_index.upsert(vectors=pinecone_vectors[i:i + batch_size])
        
        # 5. Store the complete document structure in MongoDB
        await new_document.insert()
        print(f"Successfully processed document with {len(new_document.parent_chunks)} parent chunks and {len(all_child_chunks)} child chunks.")
        return new_document

    async def answer_question(self, document_source: str, question: str) -> str:
        """High-accuracy RAG pipeline with Parent Document Retrieval."""
        document = await Document.find_one(Document.source_url == document_source)
        if not document:
            document = await self._process_new_document(document_source)

        # 1. Multi-Query Generation
        query_generation_prompt = f"Generate 3 alternative versions of this question for a document search: {question}"
        query_generation_response = await self.llm.ainvoke(query_generation_prompt)
        all_queries = [question] + [q.strip() for q in query_generation_response.content.strip().split('\n') if q.strip()]

        # 2. Hybrid Search on CHILD chunks
        all_child_chunks = [child for parent in document.parent_chunks for child in parent.children]
        all_child_texts = [chunk.text for chunk in all_child_chunks]
        
        # --- FIX: Add a check for empty documents to prevent division by zero ---
        if not all_child_texts:
            return "Error: The document is empty or contains no readable text. Cannot process the query."
        # --------------------------------------------------------------------
        
        bm25_retriever = BM25Retriever.from_texts(all_child_texts, k=15)
        
        retrieved_child_ids = set()
        pinecone_index = pinecone_client.get_index()

        for query_text in all_queries:
            # Vector search
            query_embedding = self.embeddings_model.embed_query(query_text)
            query_response = pinecone_index.query(vector=query_embedding, top_k=15, include_metadata=False)
            if query_response and query_response.get('matches'):
                for match in query_response['matches']:
                    retrieved_child_ids.add(match['id'])
            
            # Keyword search (Note: This is a simplified mapping)
            bm25_results = bm25_retriever.invoke(query_text)
            for doc in bm25_results:
                for child in all_child_chunks:
                    if child.text == doc.page_content:
                        retrieved_child_ids.add(child.chunk_id)
                        break
                        
        # 3. Retrieve PARENT Chunks
        child_to_parent_map = {
            child.chunk_id: parent.text 
            for parent in document.parent_chunks 
            for child in parent.children
        }
        
        parent_texts_to_rerank = {child_to_parent_map.get(child_id) for child_id in retrieved_child_ids if child_to_parent_map.get(child_id)}
        initial_docs = list(parent_texts_to_rerank)

        if not initial_docs:
            return "Could not find relevant information in the document."

        # 4. Re-rank the PARENT chunks
        reranked_results = self.cohere_client.rerank(
            query=question, documents=initial_docs, top_n=5, model="rerank-english-v3.0"
        )
        context_chunks = [result.document['text'] for result in reranked_results.results]
        context = "\n---\n".join(context_chunks)
        
        # 5. Generate Final Answer
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
