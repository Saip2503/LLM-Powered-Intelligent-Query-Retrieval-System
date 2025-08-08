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
        """High-accuracy configuration using a true Hybrid Parent-Child RAG pipeline."""
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        
        self.cohere_client = cohere.Client(settings.COHERE_API_KEY)

    def _get_text_from_source(self, source: str) -> str:
        """Gets text content from either a URL or a local file path."""
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
        """Processes a new document using the Parent Document chunking strategy."""
        print(f"Processing new document with Parent-Child strategy from {source_url_or_path}...")
        document_text = self._get_text_from_source(source_url_or_path)
        
        if not document_text:
             print(f"WARNING: No text could be extracted from {source_url_or_path}. Skipping.")
             empty_doc = Document(source_url=source_url_or_path, parent_chunks=[])
             await empty_doc.insert()
             return empty_doc

        parent_texts = self.parent_splitter.split_text(document_text)
        new_document = Document(source_url=source_url_or_path, parent_chunks=[])
        
        all_child_chunks = []
        for parent_text in parent_texts:
            parent_chunk = ParentChunk(text=parent_text, children=[])
            child_texts = self.child_splitter.split_text(parent_text)
            for child_text in child_texts:
                child_chunk = ChildChunk(text=child_text)
                parent_chunk.children.append(child_chunk)
                all_child_chunks.append(child_chunk)
            new_document.parent_chunks.append(parent_chunk)

        child_texts_for_embedding = [chunk.text for chunk in all_child_chunks]
        child_embeddings = self.embeddings_model.embed_documents(child_texts_for_embedding)

        pinecone_vectors = []
        for i, child_chunk in enumerate(all_child_chunks):
            pinecone_vectors.append({
                "id": child_chunk.chunk_id,
                "values": child_embeddings[i],
                "metadata": {} # No need to store text here anymore
            })

        pinecone_index = pinecone_client.get_index()
        if pinecone_vectors:
            batch_size = 100
            for i in range(0, len(pinecone_vectors), batch_size):
                pinecone_index.upsert(vectors=pinecone_vectors[i:i + batch_size])
        
        await new_document.insert()
        print(f"Successfully processed document with {len(new_document.parent_chunks)} parent chunks and {len(all_child_chunks)} child chunks.")
        return new_document

    async def answer_question(self, document_source: str, question: str) -> str:
        """High-accuracy RAG pipeline with a true Hybrid Parent-Child Retrieval."""
        document = await Document.find_one(Document.source_url == document_source)
        if not document:
            document = await self._process_new_document(document_source)

        if not document.parent_chunks:
            return "Error: The document is empty or contains no readable text. Cannot process the query."

        # 1. Multi-Query Generation
        query_generation_prompt = f"Generate 3 alternative versions of this question for a document search: {question}"
        query_generation_response = await self.llm.ainvoke(query_generation_prompt)
        all_queries = [question] + [q.strip() for q in query_generation_response.content.strip().split('\n') if q.strip()]
        
        # 2. Build lookup maps for efficient retrieval
        child_to_parent_map = {
            child.chunk_id: parent.text 
            for parent in document.parent_chunks 
            for child in parent.children
        }
        parent_texts = [parent.text for parent in document.parent_chunks]

        # 3. Perform Hybrid Search
        # 3a. Keyword search on PARENT chunks
        bm25_retriever = BM25Retriever.from_texts(parent_texts, k=10)
        keyword_retrieved_parents = {doc.page_content for query in all_queries for doc in bm25_retriever.invoke(query)}

        # 3b. Vector search on CHILD chunks
        vector_retrieved_parent_texts = set()
        pinecone_index = pinecone_client.get_index()
        query_embeddings = self.embeddings_model.embed_documents(all_queries)

        for embedding in query_embeddings:
            query_response = pinecone_index.query(vector=embedding, top_k=10, include_metadata=False)
            if query_response and query_response.get('matches'):
                for match in query_response['matches']:
                    parent_text = child_to_parent_map.get(match['id'])
                    if parent_text:
                        vector_retrieved_parent_texts.add(parent_text)
        
        # 3c. Combine results into a single pool for the re-ranker
        initial_docs = list(keyword_retrieved_parents.union(vector_retrieved_parent_texts))

        if not initial_docs:
            return "Could not find relevant information in the document."

        # 4. Re-rank the combined PARENT chunks
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
