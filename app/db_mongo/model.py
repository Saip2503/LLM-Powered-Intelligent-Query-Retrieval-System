from pydantic import BaseModel, Field
from beanie import Document as BeanieDocument, Indexed # <-- Import Indexed
from typing import List, Optional, Annotated # <-- Import Annotated
from datetime import datetime
import uuid

class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str

class Document(BeanieDocument):
    # --- THIS IS THE CHANGE ---
    # Add a unique index to the source_url field
    source_url: Annotated[str, Indexed(unique=True)]
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    chunks: Optional[List[Chunk]] = []

    class Settings:
        name = "documents"
