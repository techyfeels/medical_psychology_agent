"""
Load medical psychology dataset from HuggingFace
"""
from datasets import load_dataset
from typing import List, Dict
from medical_psychology_agent.config import Config

class MedicalDataLoader:
    """Load and prepare medical psychology dataset"""
    
    def __init__(self, dataset_name: str = "169Pi/medical_psychology"):
        self.dataset_name = dataset_name
        self.dataset = None
        
    def load(self, split: str = "train", streaming: bool = False):
        """Load dataset from HuggingFace"""
        print(f"📥 Loading dataset: {self.dataset_name}")
        
        try:
            if Config.HF_TOKEN:
                self.dataset = load_dataset(
                    self.dataset_name,
                    split=split,
                    streaming=streaming,
                    token=Config.HF_TOKEN
                )
            else:
                self.dataset = load_dataset(
                    self.dataset_name,
                    split=split,
                    streaming=streaming
                )
            
            print(f"✅ Dataset loaded successfully!")
            if not streaming:
                print(f"   Total examples: {len(self.dataset)}")
            
            return self.dataset
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def prepare_documents(self, max_samples: int = None) -> List[Dict[str, str]]:
        """
        Prepare documents for vector store ingestion
        
        Args:
            max_samples: Limit number of samples (None for all)
            
        Returns:
            List of document dictionaries with 'content' and 'metadata'
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        documents = []
        dataset_iter = iter(self.dataset)
        
        count = 0
        for idx, item in enumerate(dataset_iter):
            if max_samples and count >= max_samples:
                break
            
            # Adjust based on actual dataset structure
            # Inspect first item to understand structure
            if idx == 0:
                print(f"\n📋 Dataset structure (first item):")
                print(f"   Keys: {item.keys()}")
                print(f"   Sample: {str(item)[:200]}...\n")
            
            # Create document content
            # Adjust field names based on actual dataset structure
            content = self._create_content(item)
            
            if content:  # Only add if content exists
                doc = {
                    "content": content,
                    "metadata": {
                        "source": self.dataset_name,
                        "index": idx,
                        **self._extract_metadata(item)
                    }
                }
                documents.append(doc)
                count += 1
            
            if (count + 1) % 1000 == 0:
                print(f"   Processed {count + 1} documents...")
        
        print(f"✅ Prepared {len(documents)} documents for ingestion")
        return documents
    
    def _create_content(self, item: Dict) -> str:
        """
        Create content string from dataset item
        Adjust based on your dataset structure
        """
        # Common field names in medical datasets
        possible_fields = ['text', 'content', 'question', 'answer', 'conversation', 'prompt', 'response']
        
        content_parts = []
        
        # Try to find and combine relevant fields
        for field in possible_fields:
            if field in item and item[field]:
                content_parts.append(str(item[field]))
        
        # If conversation format (messages)
        if 'messages' in item:
            for msg in item['messages']:
                if isinstance(msg, dict):
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    content_parts.append(f"{role}: {content}")
        
        # Combine all parts
        return "\n\n".join(content_parts) if content_parts else ""
    
    def _extract_metadata(self, item: Dict) -> Dict:
        """Extract useful metadata from item"""
        metadata = {}
        
        # Add relevant metadata fields
        metadata_fields = ['category', 'topic', 'specialty', 'condition', 'type']
        
        for field in metadata_fields:
            if field in item and item[field]:
                metadata[field] = str(item[field])
        
        return metadata

if __name__ == "__main__":
    # Test the data loader
    loader = MedicalDataLoader()
    dataset = loader.load()
    
    # Prepare first 10 documents to see structure
    docs = loader.prepare_documents(max_samples=10)
    
    print(f"\n📄 Sample document:")
    print(f"Content preview: {docs[0]['content'][:300]}...")
    print(f"Metadata: {docs[0]['metadata']}")