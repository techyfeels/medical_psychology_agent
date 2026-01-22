"""Qdrant vector store setup and management."""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from medical_psychology_agent.config import Config


class VectorStoreManager:
    """Manage Qdrant vector store operations."""

    def __init__(self) -> None:
        # Validate config/env first
        Config.validate()

        self.client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            timeout=120.0,
        )

        # LangChain embeddings wrapper
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            api_key=Config.OPENAI_API_KEY,  # NOTE: langchain-openai uses `api_key`
        )

        self.collection_name = Config.QDRANT_COLLECTION_NAME

    def create_collection(self, vector_size: int = 1536, recreate: bool = False) -> None:
        """Create Qdrant collection.

        Args:
            vector_size: embedding dimension (1536 for text-embedding-3-small)
            recreate: if True, drop then re-create the collection
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists:
            if recreate:
                print(f"🗑️  Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"✅ Collection '{self.collection_name}' already exists")
                return

        print(f"📦 Creating collection: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print("✅ Collection created successfully!")

    def ingest_documents(self, documents: List[Dict], batch_size: int = 100) -> QdrantVectorStore:
        """Ingest documents into Qdrant using LangChain wrapper.

        Expected input format:
            documents = [
                {"content": "...", "metadata": {"source": "...", ...}},
                ...
            ]
        """
        if not documents:
            raise ValueError("documents is empty")

        print(f"\n🔄 Starting ingestion of {len(documents)} documents...")
        texts = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        vectorstore = QdrantVectorStore.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            collection_name=self.collection_name,
            batch_size=batch_size,
        )

        print(f"✅ Successfully ingested {len(documents)} documents!")
        return vectorstore

    def get_vectorstore(self) -> QdrantVectorStore:
        """Get an existing vectorstore instance for retrieval."""
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def get_retriever(self, k: int = 5, score_threshold: float = 0.7):
        """Get retriever with configurable parameters."""
        vectorstore = self.get_vectorstore()
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )

    def get_collection_info(self):
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            print("\n📊 Collection Info:")
            print(f"   Points count: {info.points_count}")
            print(f"   Vectors config: {info.config.params.vectors}")
            return info
        except Exception as e:
            print(f"❌ Collection not found: {e}")
            return None

    def test_search(self, query: str, k: int = 3):
        """Test search functionality."""
        print(f"\n🔎 Testing search with query: '{query}'")
        vectorstore = self.get_vectorstore()
        results = vectorstore.similarity_search_with_score(query, k=k)

        print(f"\n📋 Top {k} Results:")
        for idx, (doc, score) in enumerate(results, 1):
            print(f"\n{idx}. Score: {score:.4f}")
            print(f"   Content: {doc.page_content[:200]}...")
            print(f"   Metadata: {doc.metadata}")

        return results


if __name__ == "__main__":
    vs_manager = VectorStoreManager()
    vs_manager.get_collection_info()
    # vs_manager.test_search("What is anxiety disorder?")
