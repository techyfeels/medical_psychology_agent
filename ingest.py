"""
Script to ingest HuggingFace dataset into Qdrant
Run this once to populate your vector database
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import MedicalDataLoader
from medical_psychology_agent.vectorstore import VectorStoreManager
from config import Config

def main():
    """Main ingestion pipeline"""
    
    print("=" * 60)
    print("🚀 Medical Psychology Dataset Ingestion Pipeline")
    print("=" * 60)
    
    # Validate configuration
    Config.validate()
    Config.print_config()
    
    # Configuration
    MAX_SAMPLES = 1000  # Set to number (e.g., 1000) for testing, None for all
    RECREATE_COLLECTION = True  # Set True to recreate collection
    BATCH_SIZE = 10  # Reduced batch size to avoid timeout
    
    try:
        # Step 1: Load dataset
        print("\n📥 STEP 1: Loading dataset from HuggingFace")
        loader = MedicalDataLoader()
        dataset = loader.load()
        
        # Step 2: Prepare documents
        print(f"\n📝 STEP 2: Preparing documents")
        documents = loader.prepare_documents(max_samples=MAX_SAMPLES)
        
        if not documents:
            print("❌ No documents to ingest!")
            return
        
        # Step 3: Setup vector store
        print(f"\n🗄️  STEP 3: Setting up Qdrant collection")
        vs_manager = VectorStoreManager()
        vs_manager.create_collection(recreate=RECREATE_COLLECTION)
        
        # Step 4: Ingest documents
        print(f"\n💾 STEP 4: Ingesting documents to Qdrant")
        vectorstore = vs_manager.ingest_documents(
            documents=documents,
            batch_size=BATCH_SIZE
        )
        
        # Step 5: Verify ingestion
        print(f"\n✅ STEP 5: Verification")
        vs_manager.get_collection_info()
        
        # Test search
        print(f"\n🧪 Testing search functionality...")
        test_queries = [
            "What is depression?",
            "Apa itu gangguan kecemasan?",  # Indonesian test
            "How to treat insomnia?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            results = vs_manager.test_search(query, k=2)
        
        print(f"\n{'='*60}")
        print("✅ Ingestion completed successfully!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()