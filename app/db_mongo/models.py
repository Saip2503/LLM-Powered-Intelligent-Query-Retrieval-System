from pydantic import BaseModel, Field
from beanie import Document as BeanieDocument, Indexed
from typing import List, Optional, Annotated
from datetime import datetime
import uuid

class Chunk(BaseModel):
    """Represents a single, standard chunk of text."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str

class Document(BeanieDocument):
    """Represents a processed document with a simple list of chunks."""
    source_url: Annotated[str, Indexed(unique=True)]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    chunks: List[Chunk] = []

    class Settings:
        name = "documents"
