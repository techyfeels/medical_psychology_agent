"""
Configuration module for loading environment variables
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for all environment variables"""
    
    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "medical_psychology")
    
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Langfuse
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    
    # Cohere (optional for reranker)
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    
    # HuggingFace
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    @classmethod
    def validate(cls):
        """Validate required environment variables"""
        required = [
            "QDRANT_URL",
            "QDRANT_API_KEY",
            "OPENAI_API_KEY",
        ]
        
        missing = [var for var in required if not getattr(cls, var)]
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        print("✅ All required environment variables loaded successfully")
        
    @classmethod
    def print_config(cls):
        """Print current configuration (hiding sensitive data)"""
        print("\n📋 Current Configuration:")
        print(f"  Qdrant URL: {cls.QDRANT_URL}")
        print(f"  Collection: {cls.QDRANT_COLLECTION_NAME}")
        print(f"  LLM Model: {cls.LLM_MODEL}")
        print(f"  Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"  Langfuse: {'Enabled' if cls.LANGFUSE_SECRET_KEY else 'Disabled'}")
        print(f"  Cohere Reranker: {'Enabled' if cls.COHERE_API_KEY else 'Disabled'}")
        print()

if __name__ == "__main__":
    Config.validate()
    Config.print_config()

    