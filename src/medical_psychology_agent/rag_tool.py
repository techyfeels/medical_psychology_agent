"""
RAG tool with optional Cohere reranker and translation support for improved retrieval
"""
from langchain.tools import tool
from langchain_core.documents import Document
from typing import List, Optional
import cohere
from medical_psychology_agent.config import Config
from medical_psychology_agent.vectorstore import VectorStoreManager
from medical_psychology_agent.translator import detect_language, translate_to_english

class RAGTool:
    """RAG tool with retrieval, translation, and optional reranking"""
    
    def __init__(self, use_reranker: bool = True, use_translation: bool = True, top_k: int = 5, rerank_top_n: int = 3):
        """
        Initialize RAG tool
        
        Args:
            use_reranker: Whether to use Cohere reranker
            use_translation: Whether to translate Indonesian queries to English
            top_k: Number of documents to retrieve initially
            rerank_top_n: Number of documents to return after reranking
        """
        self.vs_manager = VectorStoreManager()
        self.retriever = self.vs_manager.get_retriever(k=top_k)
        self.use_reranker = use_reranker and Config.COHERE_API_KEY is not None
        self.use_translation = use_translation
        self.rerank_top_n = rerank_top_n
        
        if self.use_reranker:
            self.cohere_client = cohere.Client(Config.COHERE_API_KEY)
            print("✅ Cohere reranker enabled")
        else:
            self.cohere_client = None
            if use_reranker:
                print("⚠️  Cohere API key not found - reranker disabled")
        
        if self.use_translation:
            print("✅ Query translation enabled for Indonesian queries")
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for query
        
        Args:
            query: User query (can be English or Indonesian)
            
        Returns:
            List of relevant documents
        """
        # Detect language and translate if Indonesian
        search_query = query
        original_language = "english"
        
        if self.use_translation:
            detected_lang = detect_language(query)
            original_language = detected_lang
            
            if detected_lang == "indonesian":
                print(f"🌐 Detected Indonesian query, translating for better retrieval...")
                search_query = translate_to_english(query)
                print(f"   Original: {query}")
                print(f"   Translated: {search_query}")
        
        # Initial retrieval with (potentially translated) query
        documents = self.retriever.invoke(search_query)
        
        if not documents:
            print(f"⚠️  No documents found for query: {search_query}")
            return []
        
        print(f"📥 Retrieved {len(documents)} documents")
        
        # Rerank if enabled
        if self.use_reranker and len(documents) > 1:
            # Use translated query for reranking if available
            documents = self._rerank_documents(search_query, documents)
        
        return documents
    
    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents using Cohere reranker
        
        Args:
            query: Search query (already translated if needed)
            documents: Retrieved documents
            
        Returns:
            Reranked documents
        """
        try:
            # Prepare documents for reranking
            doc_texts = [doc.page_content for doc in documents]
            
            # Call Cohere rerank API
            rerank_response = self.cohere_client.rerank(
                query=query,
                documents=doc_texts,
                top_n=min(self.rerank_top_n, len(documents)),
                model="rerank-english-v3.0"
            )
            
            # Get reranked documents
            reranked_docs = []
            for result in rerank_response.results:
                reranked_docs.append(documents[result.index])
            
            print(f"🔄 Reranked to top {len(reranked_docs)} documents")
            
            return reranked_docs
            
        except Exception as e:
            print(f"⚠️  Reranking failed: {e}. Using original order.")
            return documents[:self.rerank_top_n]
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            documents: List of documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for idx, doc in enumerate(documents, 1):
            context_parts.append(f"[Document {idx}]")
            context_parts.append(doc.page_content)
            
            # Add metadata if available
            if doc.metadata:
                relevant_metadata = {k: v for k, v in doc.metadata.items() 
                                   if k in ['source', 'category', 'specialty']}
                if relevant_metadata:
                    context_parts.append(f"Metadata: {relevant_metadata}")
            
            context_parts.append("")  # Empty line between documents
        
        return "\n".join(context_parts)
    
    @tool
    def retrieve_medical_info(query: str) -> str:
        """
        Retrieve relevant medical psychology information for a given query.
        
        Use this tool when the user asks about:
        - Mental health conditions
        - Psychological disorders
        - Treatment approaches
        - Clinical information
        - Medical psychology topics
        
        Args:
            query: The user's question about medical psychology
            
        Returns:
            Relevant context from the medical knowledge base
        """
        tool_instance = RAGTool()
        documents = tool_instance.retrieve(query)
        return tool_instance.format_context(documents)

def create_rag_tool(use_reranker: bool = True, use_translation: bool = True) -> tool:
    """
    Factory function to create RAG tool
    
    Args:
        use_reranker: Whether to enable Cohere reranker
        use_translation: Whether to enable query translation
        
    Returns:
        RAG tool function
    """
    rag = RAGTool(use_reranker=use_reranker, use_translation=use_translation)
    
    @tool
    def retrieve_medical_info(query: str) -> str:
        """
        Retrieve relevant medical psychology information.
        
        Args:
            query: User's medical psychology question
            
        Returns:
            Relevant context from knowledge base
        """
        documents = rag.retrieve(query)
        return rag.format_context(documents)
    
    return retrieve_medical_info

if __name__ == "__main__":
    # Test RAG tool with translation
    print("Testing RAG Tool with Translation Support...")
    
    rag = RAGTool(use_reranker=True, use_translation=True)
    
    test_queries = [
        "What is depression?",
        "Apa itu gangguan kecemasan?",
        "Bagaimana cara mengatasi insomnia?",
        "How to treat PTSD?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        docs = rag.retrieve(query)
        context = rag.format_context(docs)
        
        print(f"\nContext preview:")
        print(context[:300])
        print("...")
        print(f"\nRetrieved {len(docs)} documents")
        print(f"{'='*60}")