# Add BackgroundTasks to your imports from fastapi
from fastapi import APIRouter, Depends, HTTPException, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings
from app.schemas.schemas import SubmissionRequest, SubmissionResponse
from app.services.document_service import document_service
from app.db_mongo.model import Document # Import the Document model

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
    background_tasks: BackgroundTasks, # Add the BackgroundTasks dependency
    token: str = Security(verify_token)
):
    """
    Processes a document and answers questions using a hybrid workflow.
    """
    # First, check if the document exists without processing it
    document = await Document.find_one(Document.source_url == request.documents)

    if not document:
        # --- SLOW PATH: NEW DOCUMENT ---
        # Trigger the slow ingestion process to run in the background
        background_tasks.add_task(
            document_service._process_new_document, 
            request.documents
        )
        # Immediately return a "processing" message
        processing_message = "The requested document is new and is being processed in the background. Please try your query again in 1-2 minutes."
        return SubmissionResponse(answers=[processing_message for _ in request.questions])

    # --- FAST PATH: DOCUMENT EXISTS ---
    # The document is already processed, so we can answer the questions directly
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
