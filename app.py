"""
Streamlit UI for Medical Psychology Agent
"""
import streamlit as st

from medical_psychology_agent.agent import MedicalPsychologyAgent
from medical_psychology_agent.config import Config

# Page configuration
st.set_page_config(
    page_title="Medical Psychology Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .assistant-message {
        background-color: #F5F5F5;
    }
    .info-box {
        padding: 1rem;
        border-left: 4px solid #1E88E5;
        background-color: #E3F2FD;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    with st.spinner("🔧 Initializing Medical Psychology Assistant..."):
        try:
            st.session_state.agent = MedicalPsychologyAgent(
                use_reranker=True,
                use_translation=True,
                use_langfuse=True,
            )
            st.session_state.agent_ready = True
        except Exception as e:
            st.error(f"❌ Error initializing agent: {e}")
            st.session_state.agent_ready = False

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 Medical Psychology Assistant")
    st.markdown("---")

    st.markdown("#### 📋 About")
    st.info(
        """
    This AI assistant helps answer questions about medical psychology,
    mental health conditions, and therapeutic approaches.

    **Powered by:**
    - OpenAI GPT-4o-mini
    - Qdrant Vector Database
    - LangGraph Supervisor Agent
    - Medical Psychology Dataset (~296k examples)
    """
    )

    st.markdown("---")
    st.markdown("#### 🌐 Language Support")
    st.success("✅ English\n\n✅ Bahasa Indonesia")

    st.markdown("---")
    st.markdown("#### ⚙️ Settings")

    with st.expander("🤖 Model Configuration"):
        st.text(f"LLM: {Config.LLM_MODEL}")
        st.text(f"Embeddings: {Config.EMBEDDING_MODEL}")
        st.text(f"Collection: {Config.QDRANT_COLLECTION_NAME}")

    with st.expander("🎯 Capabilities"):
        st.markdown(
            """
        - Mental health conditions
        - Psychological disorders
        - Treatment approaches
        - Therapeutic techniques
        - Clinical psychology
        - Evidence-based information
        """
        )

    st.markdown("---")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("#### ⚠️ Disclaimer")
    st.warning(
        """
    This assistant provides educational information only.

    **Not a substitute for professional medical advice.**

    Please consult a healthcare professional for medical concerns.
    """
    )

# Main content
st.markdown('<div class="main-header">🏥 Medical Psychology Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask me anything about mental health and psychology</div>', unsafe_allow_html=True)

# Example queries
if not st.session_state.messages:
    st.markdown("### 💡 Example Questions:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**English:**")
        if st.button("What is cognitive behavioral therapy?"):
            st.session_state.messages.append(
                {"role": "user", "content": "What is cognitive behavioral therapy?"}
            )
            st.rerun()
        if st.button("How to manage anxiety?"):
            st.session_state.messages.append(
                {"role": "user", "content": "How to manage anxiety?"}
            )
            st.rerun()

    with col2:
        st.markdown("**Bahasa Indonesia:**")
        if st.button("Apa itu gangguan depresi mayor?"):
            st.session_state.messages.append(
                {"role": "user", "content": "Apa itu gangguan depresi mayor?"}
            )
            st.rerun()
        if st.button("Bagaimana cara mengatasi insomnia?"):
            st.session_state.messages.append(
                {"role": "user", "content": "Bagaimana cara mengatasi insomnia?"}
            )
            st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("ℹ️ Response Details"):
                st.json(message["metadata"])

# Chat input
if prompt := st.chat_input("Ask your question here... (English or Bahasa Indonesia)"):
    if not st.session_state.agent_ready:
        st.error("⚠️ Agent not initialized. Please check your configuration.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.agent.query(prompt)

                    st.markdown(response["answer"])

                    if response["context_used"]:
                        st.info("📚 Answer generated using medical psychology knowledge base (RAG)")
                    else:
                        st.caption("💬 General response (no document retrieval)")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response["answer"],
                            "metadata": {
                                "agent_used": response.get("agent_used"),
                                "has_context": response.get("context_used") is not None,
                            },
                        }
                    )

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666;'>
    <small>
    Built with LangChain, LangGraph, Qdrant & Streamlit |
    Powered by OpenAI GPT-4o-mini |
    Medical Psychology Dataset from HuggingFace
    </small>
</div>
""",
    unsafe_allow_html=True,
)
