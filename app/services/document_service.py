import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.db_mongo.models import Document, Chunk
from app.vector_db.pinecone_client import pinecone_client
from app.core.config import settings
from unstructured.partition.pdf import partition_pdf # Import the new parser

class DocumentService:
    def __init__(self):
        """Final, high-accuracy RAG configuration with Layout-Aware Parsing."""
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

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
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(source, headers=headers, timeout=60)
                response.raise_for_status()
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
        """Processes a new document using the improved parsing and chunking."""
        print(f"Processing new document from {source_url_or_path}...")
        document_text = self._get_text_from_source(source_url_or_path)
        
        if not document_text:
             print(f"WARNING: No text could be extracted. Skipping.")
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
                "id": chunk.chunk_id, "values": chunk_embeddings[i], "metadata": {"text": chunk.text}
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
        """A simple and direct RAG pipeline with high-quality data."""
        document = await Document.find_one(Document.source_url == document_source)
        if not document:
            document = await self._process_new_document(document_source)

        if not document.chunks:
            return "Error: The document is empty or contains no readable text."

        # 1. Vector Search
        pinecone_index = pinecone_client.get_index()
        query_embedding = self.embeddings_model.embed_query(question)
        query_response = pinecone_index.query(vector=query_embedding, top_k=15, include_metadata=True)
        
        if not query_response or not query_response.matches:
            return "Could not find relevant information in the document."

        context_chunks = [match.metadata['text'] for match in query_response.matches if match.metadata and 'text' in match.metadata]
        context = "\n---\n".join(context_chunks)
        
        # 2. Generate Final Answer
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
