# Remove BackgroundTasks from your imports from fastapi
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings
from app.schemas.schemas import SubmissionRequest, SubmissionResponse
from app.services.document_service import document_service
# You no longer need to import the Document model here

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
    # The BackgroundTasks dependency is removed
    token: str = Security(verify_token)
):
    """
    Processes a document and answers questions in a single, synchronous request.
    """
    # The logic is now simplified back to its original form.
    # The document_service.answer_question method will handle both new and existing documents.
    answers = []
    for question in request.questions:
        try:
            answer = await document_service.answer_question(
                document_source=request.documents,
                question=question
            )
            answers.append(answer)
        except Exception as e:
            answers.append(f"An error occurred while processing this question: {str(e)}")
            
    return SubmissionResponse(answers=answers)
