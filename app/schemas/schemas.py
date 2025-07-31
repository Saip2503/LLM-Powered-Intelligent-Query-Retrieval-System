from pydantic import BaseModel, Field
from typing import List

class SubmissionRequest(BaseModel):
    """
    Defines the structure for the incoming request payload.
    """
    documents: str = Field(..., description="URL to the document (e.g., PDF) to be processed.")
    questions: List[str] = Field(..., description="A list of questions to ask about the document.")

class SubmissionResponse(BaseModel):
    """
    Defines the structure for the API's JSON response.
    """
    answers: List[str] = Field(..., description="A list of generated answers, corresponding to the input questions.")
