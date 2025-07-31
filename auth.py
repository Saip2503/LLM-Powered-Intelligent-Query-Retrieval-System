# auth.py

import os
import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# This scheme will look for a token in the "Authorization: Bearer <token>" header
bearer_scheme = HTTPBearer()

# Retrieve the expected token from environment variables
EXPECTED_TOKEN = os.getenv("EXPECTED_BEARER_TOKEN")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """
    A dependency function to validate the bearer token.

    This function securely compares the provided token with the one stored
    in the environment variables.

    Raises:
        HTTPException(500): If the server is misconfigured (token not set).
        HTTPException(403): If the provided token is invalid.
    """
    if not EXPECTED_TOKEN:
        # Server-side check to ensure the environment variable is set.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: Token not set."
        )

    # Use a secure, constant-time comparison to prevent timing attacks.
    is_correct_token = secrets.compare_digest(
        credentials.credentials, EXPECTED_TOKEN
    )

    if not (credentials.scheme == "Bearer" and is_correct_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing authentication token"
        )
    
    # You can return the token or a success indicator if needed elsewhere
    return credentials.credentials