import asyncio
import os
from app.services.document_service import DocumentService
from app.db_mongo.mongo_client import init_db
from app.vector_db.pinecone_client import pinecone_client

# --- Updated list of all documents to be ingested ---
DOCUMENT_SOURCES = [
    # Local Files
    "data/doc1.pdf",
    "data/BAJHLIP23020V012223.pdf",
    "data/CHOTGDP23004V012223.pdf",
    "data/EDLHLGA23009V012223.pdf",
    "data/HDFHLIP23024V072223.pdf",
    "data/ICIHLIP22012V012223.pdf",
    "data/Arogya Sanjeevani Policy - CIN - U10200WB1906GOI001713 1.pdf",
    "data/Family Medicare Policy (UIN- UIIHLIP22070V042122) 1.pdf",
    "data/policy.pdf",
    "data/principia_newton.pdf",
    "data/Super_Splendor_(Feb_2023).pdf",
    
    # URLs
    "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
    "https://hackrx.blob.core.windows.net/assets/Happy%20Family%20Floater%20-%202024%20OICHLIP25046V062425%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A24%3A30Z&se=2026-08-01T17%3A24%3A00Z&sr=b&sp=r&sig=VNMTTQUjdXGYb2F4Di4P0zNvmM2rTBoEHr%2BnkUXIqpQ%3D",
]

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
