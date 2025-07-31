from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings

class PineconeClient:
    """
    Manages the connection to the Pinecone vector database.
    """
    def __init__(self):
        self.client = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.index = None

    def connect(self):
        """
        Initializes the connection and creates the index if it doesn't exist.
        """
        # Set dimension based on the embedding model
        # Gemini's text-embedding-004 model has a dimension of 768
        dimension = 768 
        
        if self.index_name not in self.client.list_indexes().names():
            self.client.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine", # Cosine similarity is great for semantic search
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        self.index = self.client.Index(self.index_name)
        print(f"Pinecone index '{self.index_name}' is ready.")

    def get_index(self):
        """
        Returns the connected Pinecone index object.
        """
        if not self.index:
            self.connect()
        return self.index

# Global instance of the Pinecone client
pinecone_client = PineconeClient()
