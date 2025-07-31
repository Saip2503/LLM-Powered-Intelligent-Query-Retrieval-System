from pydantic import BaseModel, Field
from beanie import Document as BeanieDocument
from typing import List, Optional
from datetime import datetime
import uuid

class Chunk(BaseModel):
    """
    Represents a single chunk of text embedded within a Document.
    This is a Pydantic model, not a standalone Beanie document.
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str

class Document(BeanieDocument):
    """
    Represents a processed document stored in MongoDB.
    This is a Beanie Document model, which corresponds to a collection.
    """
    source_url: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    chunks: Optional[List[Chunk]] = []

    class Settings:
        name = "documents" # This will be the collection name in MongoDB
