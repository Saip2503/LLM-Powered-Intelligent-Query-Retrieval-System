from pydantic import BaseModel, Field
from beanie import Document as BeanieDocument, Indexed
from typing import List, Optional, Annotated
from datetime import datetime
import uuid

class ChildChunk(BaseModel):
    """Represents a small, precise chunk for vector search."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str

class ParentChunk(BaseModel):
    """Represents a larger, context-rich chunk to be sent to the LLM."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    children: List[ChildChunk] = []

class Document(BeanieDocument):
    """Represents a processed document, now containing parent chunks."""
    source_url: Annotated[str, Indexed(unique=True)]
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow)
    parent_chunks: List[ParentChunk] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    class Settings:
        name = "documents"
