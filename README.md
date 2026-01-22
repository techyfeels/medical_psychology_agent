# Medical Psychology Agent

An AI-powered **medical psychology chatbot** built using **Retrieval Augmented Generation (RAG)** and a **multi-agent supervisor architecture**.

The system intelligently decides whether a user query should be answered using **retrieved medical documents** or a **direct conversational response**, ensuring accuracy, transparency, and reduced hallucination.

---

## Features

- **Bilingual Support**  
  English & Bahasa Indonesia (automatic detection + translation for retrieval)

- **Multi-Agent Architecture (LangGraph)**  
  - **Supervisor Agent** decides query routing  
  - **Retrieval Agent** answers using RAG  
  - **Direct Answer Agent** handles greetings & small talk  

- **Hybrid Supervisor Logic**
  - Rule-based routing for medical keywords (guaranteed RAG)
  - LLM-based reasoning for ambiguous queries

- **RAG System (Verified)**
  - Vector search using **Qdrant Cloud**
  - Medical psychology dataset from HuggingFace
  - Answers explicitly marked as **RAG-based** in the UI

- **Reranking**
  - Optional Cohere reranker for higher retrieval accuracy

- **Chat History Memory**
  - Maintains recent conversation context (minimum 3 turns)

- **Prompt Monitoring**
  - Integrated with **Langfuse** for prompt & trace tracking
  - Automatic fallback to local prompts if Langfuse fails

- **User-Friendly UI**
  - Streamlit-based web interface
  - Clear distinction between RAG and non-RAG responses

---

## Architecture

User Query
↓
Supervisor Agent (LangGraph)
↓
┌───────────────────────┬────────────────────────┐
│ │ │
Direct Answer Agent Retrieval Agent (RAG) │
(Small talk / greeting) │ │
↓ │
Vector Search (Qdrant Cloud) │
↓ │
(Optional) Reranker (Cohere) │
↓ │
Context-aware Answer │


---

## RAG Transparency

The application **explicitly indicates** when a response is generated using document retrieval:

- 📚 *“Answer generated using medical psychology knowledge base (RAG)”*
- 💬 *“General response (no document retrieval)”*

This ensures the user (and evaluator) can clearly distinguish:
- RAG-based answers  
- Non-RAG conversational replies  

---

## Tech Stack

- **LLM**: OpenAI GPT-4o-mini  
- **Frameworks**: LangChain, LangGraph  
- **Vector Database**: Qdrant Cloud  
- **Reranker**: Cohere (optional)  
- **Prompt Monitoring**: Langfuse  
- **UI**: Streamlit  
- **Dataset**:  
  - HuggingFace: `169Pi/medical_psychology`

---

## Deployment

The application is deployed as a **public Streamlit app** and can be accessed via:


---

## Disclaimer

This chatbot provides **educational information only** and is **not a substitute for professional medical advice**.  
For diagnosis or treatment, please consult a qualified healthcare professional.
