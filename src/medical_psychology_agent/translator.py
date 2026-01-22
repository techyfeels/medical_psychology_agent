"""
Language detection and translation utilities for bilingual support
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from medical_psychology_agent.config import Config
import re

class LanguageHandler:
    """Handle language detection and translation"""
    
    def __init__(self):
        """Initialize language handler with LLM for translation"""
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=0
        )
    
    def detect_language(self, text: str) -> str:
        """
        Detect if text is Indonesian or English
        
        Args:
            text: Input text to detect
            
        Returns:
            'indonesian' or 'english'
        """
        # Indonesian common words and patterns
        indonesian_keywords = [
            'apa', 'bagaimana', 'mengapa', 'kenapa', 'kapan', 'dimana', 'siapa', 
            'yang', 'dengan', 'untuk', 'dari', 'di', 'ke', 'ini', 'itu', 
            'dan', 'atau', 'adalah', 'ada', 'akan', 'saya', 'kamu', 'mereka',
            'gangguan', 'gejala', 'cara', 'mengatasi', 'penyakit', 'terapi',
            'kesehatan', 'mental', 'psikologi', 'dokter', 'obat'
        ]
        
        # Convert to lowercase and split
        words = text.lower().split()
        
        # Count Indonesian keywords
        indo_count = sum(1 for word in words if word in indonesian_keywords)
        
        # If more than 20% of words are Indonesian keywords, classify as Indonesian
        threshold = len(words) * 0.2
        
        detected_lang = "indonesian" if indo_count >= threshold or indo_count >= 2 else "english"
        
        return detected_lang
    
    def translate_to_english(self, text: str) -> str:
        """
        Translate Indonesian text to English for better retrieval
        
        Args:
            text: Indonesian text to translate
            
        Returns:
            English translation
        """
        try:
            system_message = SystemMessage(content="""You are a medical translator specializing in psychology and mental health terminology.
            
Your task: Translate Indonesian medical/psychology queries to English accurately.

Guidelines:
- Maintain medical terminology precision
- Keep the query structure and intent
- Only output the English translation, nothing else
- No explanations or additional text""")
            
            human_message = HumanMessage(content=f"Translate to English:\n\n{text}")
            
            response = self.llm.invoke([system_message, human_message])
            translation = response.content.strip()
            
            # Remove any quotation marks that might be added
            translation = translation.strip('"\'')
            
            return translation
            
        except Exception as e:
            print(f"⚠️ Translation failed: {e}. Using original query.")
            return text
    
    def should_translate(self, text: str) -> bool:
        """
        Determine if text should be translated for retrieval
        
        Args:
            text: Input text
            
        Returns:
            True if should translate, False otherwise
        """
        return self.detect_language(text) == "indonesian"

# Convenience functions for backward compatibility
_handler = None

def get_handler():
    """Get or create singleton language handler"""
    global _handler
    if _handler is None:
        _handler = LanguageHandler()
    return _handler

def detect_language(text: str) -> str:
    """
    Detect language of text
    
    Args:
        text: Input text
        
    Returns:
        'indonesian' or 'english'
    """
    handler = get_handler()
    return handler.detect_language(text)

def translate_to_english(text: str) -> str:
    """
    Translate Indonesian to English
    
    Args:
        text: Indonesian text
        
    Returns:
        English translation
    """
    handler = get_handler()
    return handler.translate_to_english(text)

def should_translate(text: str) -> bool:
    """
    Check if text should be translated
    
    Args:
        text: Input text
        
    Returns:
        True if should translate
    """
    handler = get_handler()
    return handler.should_translate(text)

if __name__ == "__main__":
    # Test language detection and translation
    test_cases = [
        "What is depression?",
        "Apa itu gangguan kecemasan?",
        "Bagaimana cara mengatasi insomnia?",
        "How to treat anxiety disorder?",
        "Apa gejala depresi mayor?",
        "What are the symptoms of PTSD?",
    ]
    
    handler = LanguageHandler()
    
    print("🧪 Testing Language Detection & Translation\n")
    print("="*60)
    
    for text in test_cases:
        print(f"\nOriginal: {text}")
        detected = handler.detect_language(text)
        print(f"Detected: {detected}")
        
        if detected == "indonesian":
            translated = handler.translate_to_english(text)
            print(f"Translated: {translated}")
        
        print("-"*60)