import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-index")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- TEST QUESTION ---
# Use a specific question from the original set
TEST_QUESTION = "What is the waiting period for cataract surgery?"

def run_test():
    """
    Connects to Pinecone, embeds a question, and retrieves the raw search results.
    """
    print("--- Initializing Clients ---")
    if not all([PINECONE_API_KEY, GOOGLE_API_KEY]):
        print("ERROR: Make sure PINECONE_API_KEY and GOOGLE_API_KEY are set in your .env file.")
        return

    try:
        # 1. Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        print(f"Successfully connected to Pinecone index '{PINECONE_INDEX_NAME}'.")

        # 2. Initialize the exact same embedding model used during ingestion
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        print("Successfully initialized Google Embedding model.")

        # 3. Embed the test question
        print(f"\n--- Embedding Question ---")
        print(f"Question: {TEST_QUESTION}")
        question_embedding = embeddings_model.embed_query(TEST_QUESTION)
        print("Embedding created successfully.")

        # 4. Query Pinecone
        print("\n--- Querying Pinecone ---")
        query_response = index.query(
            vector=question_embedding,
            top_k=5, # Get the top 5 matches
            include_metadata=True
        )
        print("Query sent successfully.")

        # 5. Print the Raw Results
        print("\n--- RAW RETRIEVAL RESULTS ---")
        if query_response and query_response.get('matches'):
            for i, match in enumerate(query_response['matches']):
                print(f"\n--- Result {i+1} (Score: {match.get('score', 'N/A')}) ---")
                # The metadata should contain the original text chunk
                retrieved_text = match.get('metadata', {}).get('text', 'No text metadata found.')
                print(retrieved_text)
        else:
            print("!!! NO MATCHES FOUND IN PINECONE !!!")
        
        print("\n--- Test Complete ---")

    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    run_test()

