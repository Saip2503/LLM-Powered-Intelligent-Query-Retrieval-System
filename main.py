# main.py

from fastapi import FastAPI, Depends
from auth import verify_token # Import the dependency
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Define your Pydantic models
class SubmissionRequest(BaseModel):
    documents: str
    questions: List[str]

class SubmissionResponse(BaseModel):
    answers: List[str]

@app.post(
    "/hackrx/run",
    response_model=SubmissionResponse,
    # This line protects the endpoint.
    dependencies=[Depends(verify_token)] 
)
async def run_submission(request: SubmissionRequest):
    """
    This function will only run if verify_token succeeds.
    """
    # Your main application logic (Steps 2-7) goes here.
    # For this example, we'll return a dummy response.
    
    dummy_answers = [f"Answer to '{q[:30]}...'" for q in request.questions]
    
    return SubmissionResponse(answers=dummy_answers)