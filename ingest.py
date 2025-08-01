import asyncio
import os
from app.services.document_service import DocumentService
from app.db_mongo.mongo_client import init_db
from app.vector_db.pinecone_client import pinecone_client

# --- Add your 5 PDF document URLs or local file paths here ---
# Option 1: URLs
DOCUMENT_SOURCES = [
    "data/doc1.pdf",
    "data/BAJHLIP23020V012223.pdf",
    "data/CHOTGDP23004V012223.pdf",
    "data/EDLHLGA23009V012223.pdf",
    "data/HDFHLIP23024V072223.pdf",
    "data/ICIHLIP22012V012223.pdf"    
]

# Option 2: Local Files (place them in a 'data' folder)
# DOCUMENT_SOURCES = [
#     "data/doc1.pdf",
#     "data/doc2.pdf",
# ]

async def main():
    """
    Main function to initialize services and process documents.
    """
    print("Initializing database and Pinecone connections...")
    await init_db()
    pinecone_client.connect()
    
    # Instantiate the service that contains our processing logic
    service = DocumentService()
    
    print(f"Starting ingestion for {len(DOCUMENT_SOURCES)} documents...")
    
    for source in DOCUMENT_SOURCES:
        print(f"\n--- Processing: {source} ---")
        try:
            # We reuse the same service logic that the API uses.
            # It will check if the document exists and process it if it's new.
            await service._process_new_document(source)
        except Exception as e:
            print(f"Failed to process {source}. Error: {e}")
            
    print("\n--- Ingestion Complete ---")

if __name__ == "__main__":
    # This allows us to run the async main function from the command line
    asyncio.run(main())
