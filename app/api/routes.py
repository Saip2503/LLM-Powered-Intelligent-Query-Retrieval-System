from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.api_core import exceptions as google_exceptions
from app.core.config import settings
from app.schemas.schemas import SubmissionRequest, SubmissionResponse
from app.services.document_service import document_service

router = APIRouter()
bearer_scheme = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """Dependency to validate the bearer token."""
    if credentials.scheme != "Bearer" or credentials.credentials != settings.EXPECTED_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing authentication token")
    return credentials.credentials

@router.post(
    "/hackrx/run",
    response_model=SubmissionResponse,
    summary="Run Submissions",
    tags=["HackRx"]
)
async def run_submission(
    request: SubmissionRequest,
    token: str = Security(verify_token)
):
    """
    Processes a document and answers questions using MongoDB, Pinecone, and Gemini.
    """
    answers = []
    for question in request.questions:
        try:
            answer = await document_service.answer_question(
                document_source=request.documents,
                question=question
            )
            answers.append(answer)
        except Exception as e:
            # --- NEW, MORE DETAILED ERROR HANDLING ---
            error_message = f"An unexpected error occurred: {str(e)}"
            
            # Check if it's a Google API error and extract details
            if isinstance(e, google_exceptions.GoogleAPICallError):
                error_message = f"Google API Error: {e.message}"
            
            # Add more specific checks for other libraries if needed
            
            print(f"ERROR processing question '{question}': {error_message}")
            answers.append(error_message)
            
    return SubmissionResponse(answers=answers)
