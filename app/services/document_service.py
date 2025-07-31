import fitz
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.db_mongo.model import Document, Chunk
from app.vector_db.pinecone_client import pinecone_client

class DocumentService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    def _download_and_parse_pdf(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with fitz.open(stream=response.content, filetype="pdf") as doc:
                return "".join(page.get_text() for page in doc)
        except Exception as e:
            print(f"Error downloading or parsing PDF: {e}")
            raise

    async def _process_new_document(self, url: str) -> Document:
        print(f"Processing new document from {url}...")
        document_text = self._download_and_parse_pdf(url)
        
        # 1. Chunk the text
        text_chunks = self.text_splitter.split_text(document_text)
        
        # 2. Create MongoDB document and embedded chunk models
        document_chunks = [Chunk(text=t) for t in text_chunks]
        new_document = Document(source_url=url, chunks=document_chunks)
        
        # 3. Create embeddings for Pinecone
        chunk_texts_for_embedding = [chunk.text for chunk in document_chunks]
        chunk_embeddings = self.embeddings_model.embed_documents(chunk_texts_for_embedding)
        
        # 4. Prepare vectors for Pinecone upsert
        pinecone_vectors = []
        for i, chunk in enumerate(document_chunks):
            pinecone_vectors.append({
                "id": chunk.chunk_id, # Link using the unique chunk ID
                "values": chunk_embeddings[i],
                "metadata": {"text": chunk.text}
            })

        # 5. Upsert to Pinecone
        pinecone_index = pinecone_client.get_index()
        if pinecone_vectors:
            pinecone_index.upsert(vectors=pinecone_vectors)
        
        # 6. Insert the complete document with embedded chunks into MongoDB
        await new_document.insert()
        
        print(f"Successfully processed and stored new document with {len(text_chunks)} chunks.")
        return new_document

    async def answer_question(self, document_url: str, question: str) -> str:
        # Check if the document exists in MongoDB, otherwise process and store it.
        document = await Document.find_one(Document.source_url == document_url)
        if not document:
            document = await self._process_new_document(document_url)

        # 1. Embed the user's question
        question_embedding = self.embeddings_model.embed_query(question)

        # 2. Query Pinecone to retrieve relevant chunks
        pinecone_index = pinecone_client.get_index()
        query_response = pinecone_index.query(
            vector=question_embedding,
            top_k=5,
            include_metadata=True
        )
        
        context = "\n---\n".join([match['metadata']['text'] for match in query_response['matches']])
        
        if not context:
            return "Could not find relevant information in the document to answer the question."

        # 3. Generate the answer with Gemini
        prompt = f"""
        You are an expert assistant parsing documents. Answer the question based ONLY on the provided context.
        If the answer is not found, state that "The information is not available in the provided document."
        Provide a direct and concise answer.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        response = await self.llm.ainvoke(prompt)
        return response.content

document_service = DocumentService()
