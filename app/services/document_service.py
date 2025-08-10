import requests
import cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.db_mongo.models import Document, ParentChunk, ChildChunk
from app.vector_db.pinecone_client import pinecone_client
from app.core.config import settings
from unstructured.partition.pdf import partition_pdf

class DocumentService:
    def __init__(self):
        """High-accuracy configuration using Parent Document chunking."""
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # Splitter for large, context-rich parent chunks
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        # Splitter for small, precise child chunks for vector search
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        
        self.cohere_client = cohere.Client(settings.COHERE_API_KEY)

    def _get_text_from_source(self, source: str) -> str:
        """
        Gets structured text content from a URL or local file using unstructured.
        This method now extracts tables and converts them to markdown.
        """
        print(f"Performing layout-aware parsing on: {source}")
        filepath = source
        try:
            if source.startswith("http://") or source.startswith("https://"):
                # unstructured needs a local file, so we download it first
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(source, headers=headers, timeout=60)
                response.raise_for_status()
                # Use a temporary file to store the downloaded PDF
                filepath = "temp_document.pdf"
                with open(filepath, "wb") as f:
                    f.write(response.content)

            # Use unstructured to partition the PDF, extracting text, tables, etc.
            elements = partition_pdf(
                filename=filepath,
                strategy="fast", 
                infer_table_structure=True, # Crucial for table extraction
                extract_images_in_pdf=False
            )
            
            # Convert extracted elements (including tables as markdown) into a single text string
            return "\n\n".join([str(el) for el in elements])

        except Exception as e:
            print(f"Error during layout-aware parsing for {source}: {e}")
            raise

    async def _process_new_document(self, source_url_or_path: str) -> Document:
        """Processes a new document using the Parent Document chunking strategy."""
        print(f"Processing new document with Parent-Child strategy from {source_url_or_path}...")
        document_text = self._get_text_from_source(source_url_or_path)
        
        if not document_text:
             print(f"WARNING: No text could be extracted. Skipping.")
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
                "metadata": {} # No need to store text, it's in MongoDB
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
        """High-accuracy RAG pipeline with Parent Document Retrieval and Re-ranking."""
        document = await Document.find_one(Document.source_url == document_source)
        if not document:
            document = await self._process_new_document(document_source)

        if not document.parent_chunks:
            return "Error: The document is empty or contains no readable text. Cannot process the query."

        # 1. Vector Search on CHILD chunks
        retrieved_child_ids = set()
        pinecone_index = pinecone_client.get_index()
        query_embedding = self.embeddings_model.embed_query(question)
        query_response = pinecone_index.query(vector=query_embedding, top_k=10, include_metadata=False)
        if query_response and query_response.matches:
            for match in query_response.matches:
                retrieved_child_ids.add(match.id)
                        
        # 2. Retrieve corresponding PARENT Chunks
        child_to_parent_map = {
            child.chunk_id: parent.text 
            for parent in document.parent_chunks 
            for child in parent.children
        }
        
        context_chunks = {child_to_parent_map.get(child_id) for child_id in retrieved_child_ids if child_to_parent_map.get(child_id)}
        context = "\n---\n".join(list(context_chunks))

        if not context:
            return "Could not find relevant information in the document."
        
        # 3. Generate Final Answer with the high-quality parent context
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
