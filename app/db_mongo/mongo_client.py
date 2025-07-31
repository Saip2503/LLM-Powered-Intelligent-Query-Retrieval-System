from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.core.config import settings
from .model import Document  # Import your Beanie models

async def init_db():
    """
    Initializes the database connection and Beanie ODM.
    This is called once during application startup.
    """
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    database = client[settings.MONGODB_DB_NAME]
    
    await init_beanie(
        database=database,
        document_models=[Document] # List all your Beanie Document models here
    )
    print("Beanie ODM has been initialized.")