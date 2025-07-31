import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Manages application settings and secrets."""
    PROJECT_NAME: str = "Intelligent Query–Retrieval System (MongoDB Edition)"
    API_V1_STR: str = "/api/v1"
    EXPECTED_BEARER_TOKEN: str = os.getenv("EXPECTED_BEARER_TOKEN", "your-secret-token")

    # --- MongoDB ---
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://user:password@host:port")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "hackrxdb")

    # --- Vector DB ---
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "hackrx-index")

    # --- LLM ---
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
