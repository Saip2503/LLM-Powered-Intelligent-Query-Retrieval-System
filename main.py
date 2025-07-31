from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.routes import router as api_router
from app.core.config import settings
from app.vector_db.pinecone_client import pinecone_client
from app.db_mongo.mongo_client import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    print("Application startup...")
    # Connect to Pinecone
    pinecone_client.connect()
    # Initialize Beanie and connect to MongoDB
    await init_db()
    yield
    print("Application shutdown...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to the {settings.PROJECT_NAME}"}
