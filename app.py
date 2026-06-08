"""MindPulse — Mental Health Knowledge & Referral Platform"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Load Streamlit Cloud secrets into env vars (fallback when .env not present)
try:
    import streamlit as st
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

import streamlit as st
from medical_psychology_agent.agent import MedicalPsychologyAgent
from medical_psychology_agent.config import Config

st.set_page_config(
    page_title="MindPulse",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header { font-size:2.2rem; font-weight:bold; color:#1E88E5; text-align:center; margin-bottom:.5rem; }
    .sub-header  { font-size:1.1rem; color:#666; text-align:center; margin-bottom:1.5rem; }
    .agent-badge { display:inline-block; padding:2px 10px; border-radius:12px; font-size:.75rem; font-weight:600; margin-top:6px; }
    .badge-retrieval     { background:#E3F2FD; color:#1565C0; }
    .badge-recommendation{ background:#E8F5E9; color:#2E7D32; }
    .badge-direct        { background:#F5F5F5; color:#555; }
    .badge-crisis        { background:#FFEBEE; color:#B71C1C; }
    .badge-referral      { background:#F3E5F5; color:#6A1B9A; }
    .source-card { border-left:3px solid #90CAF9; padding:8px 12px; background:#FAFAFA; margin-bottom:6px; border-radius:4px; font-size:.85rem; }
    .therapist-card { border:1px solid #E0E0E0; border-radius:8px; padding:14px 16px; margin-bottom:10px; background:#FAFAFA; }
    .therapist-name { font-weight:700; font-size:1rem; color:#1A237E; }
    .therapist-title { font-size:.85rem; color:#555; margin-bottom:6px; }
    .spec-chip { display:inline-block; background:#EDE7F6; color:#4527A0; border-radius:10px; padding:2px 8px; font-size:.75rem; margin:2px; }
    .contact-row { font-size:.85rem; margin-top:8px; color:#333; }
    .provider-sub { font-size:.82rem; color:#444; margin-top:4px; border-left:2px solid #CE93D8; padding-left:8px; }
    .stat-value  { font-size:1.4rem; font-weight:700; color:#1E88E5; }
    .stat-label  { font-size:.75rem; color:#888; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Session state init
# ------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "stats" not in st.session_state:
    st.session_state.stats = {
        "total": 0,
        "retrieval": 0,
        "recommendation": 0,
        "direct": 0,
        "crisis": 0,
        "referral": 0,
        "lang_en": 0,
        "lang_id": 0,
        "feedback_pos": 0,
        "feedback_neg": 0,
    }

if "agent" not in st.session_state:
    with st.spinner("💚 Starting MindPulse..."):
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

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 💚 MindPulse")
    st.markdown("---")

    # --- Session Analytics ---
    st.markdown("#### 📊 Session Analytics")
    stats = st.session_state.stats
    total = stats["total"] or 1  # avoid div by zero

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"<div class='stat-value'>{stats['total']}</div><div class='stat-label'>Questions</div>", unsafe_allow_html=True)
    with col_b:
        rag_pct = round((stats["retrieval"] + stats["recommendation"]) / total * 100)
        st.markdown(f"<div class='stat-value'>{rag_pct}%</div><div class='stat-label'>RAG answers</div>", unsafe_allow_html=True)

    if stats["total"] > 0:
        st.markdown(
            f"""
            <div style='font-size:.8rem; margin-top:8px; color:#555;'>
            📚 Retrieval: {stats['retrieval']} &nbsp;|&nbsp;
            💡 Reco: {stats['recommendation']}<br>
            📋 Referral: {stats['referral']} &nbsp;|&nbsp;
            🆘 Crisis: {stats['crisis']}<br>
            💬 Direct: {stats['direct']}<br>
            🌐 EN: {stats['lang_en']} &nbsp;|&nbsp; ID: {stats['lang_id']}<br>
            👍 {stats['feedback_pos']} &nbsp;|&nbsp; 👎 {stats['feedback_neg']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- About ---
    st.markdown("#### 📋 About")
    st.info(
        """
    Mental health knowledge platform backed by a
    verified medical psychology dataset of 296k+ entries.

    **Powered by:**
    - OpenAI GPT-4o-mini
    - Qdrant Vector Database
    - LangGraph Multi-Agent
    - Cohere Reranker
    - Langfuse Monitoring
    - Medical Psychology Dataset (HuggingFace)
    """
    )

    st.markdown("---")
    st.markdown("#### 🤖 Agent Architecture")
    st.markdown(
        """
    ```
    Supervisor
    ├── 📚 Retrieval    (RAG)
    ├── 💡 Recommendation (RAG + Steps)
    ├── 🩺 Provider Dir  (VA/DC Providers)
    ├── 💬 Direct       (Small talk)
    └── 🆘 Crisis       (Emergency)
    ```
    """
    )

    st.markdown("---")
    st.markdown("#### 🌐 Language Support")
    st.success("✅ English\n\n✅ Bahasa Indonesia")

    st.markdown("---")
    with st.expander("⚙️ Model Configuration"):
        st.text(f"LLM: {Config.LLM_MODEL}")
        st.text(f"Embeddings: {Config.EMBEDDING_MODEL}")
        st.text(f"Collection: {Config.QDRANT_COLLECTION_NAME}")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.stats = {k: 0 for k in st.session_state.stats}
        st.rerun()

    st.markdown("---")
    st.warning(
        "⚠️ **Disclaimer:** Educational information only. Not a substitute for professional medical advice."
    )

# ------------------------------------------------------------------
# Main content
# ------------------------------------------------------------------

st.markdown('<div class="main-header">💚 MindPulse</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Your trusted mental health knowledge & referral platform</div>',
    unsafe_allow_html=True,
)

# --- Example questions (shown when chat is empty) ---
if not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    col1, col2 = st.columns(2)

    examples_en = [
        ("What is cognitive behavioral therapy?", "📚"),
        ("What should I do to manage anxiety?", "💡"),
    ]
    examples_id = [
        ("Apa itu gangguan depresi mayor?", "📚"),
        ("Bagaimana cara mengatasi insomnia?", "💡"),
    ]

    with col1:
        st.markdown("**English**")
        for text, icon in examples_en:
            if st.button(f"{icon} {text}", key=f"ex_{text[:20]}"):
                st.session_state.pending_prompt = text

    with col2:
        st.markdown("**Bahasa Indonesia**")
        for text, icon in examples_id:
            if st.button(f"{icon} {text}", key=f"ex_{text[:20]}"):
                st.session_state.pending_prompt = text

# ------------------------------------------------------------------
# Chat display
# ------------------------------------------------------------------

def render_therapist_cards(providers: list):
    """Render structured therapist/clinic cards."""
    if not providers:
        return
    st.markdown("#### 🏥 Recommended Providers — Virginia & Washington DC")
    for p in providers:
        name = p.get("name", "Unknown")
        ptype = p.get("type", "Provider")
        description = p.get("description", "")
        specialties = p.get("specialties", [])
        therapies = p.get("therapies", [])
        contact = p.get("contact", {})
        phone = contact.get("phone", "")
        website = contact.get("website") or contact.get("directory_profile", "")
        contact_form = contact.get("contact_form") or contact.get("booking", "")
        ages = p.get("ages_served", "")
        session_types = p.get("session_types", [])
        languages = p.get("languages", [])
        insurance = p.get("insurance", [])
        providers_list = p.get("providers", [])
        accepting = p.get("accepting_patients", True)

        # Single vs multi-location
        location = p.get("location", {})
        locations = p.get("locations", [])
        if location and not locations:
            loc_str = f"{location.get('address','')}, {location.get('city','')}, {location.get('state','')}"
        elif locations:
            loc_str = " · ".join(
                f"{l.get('city','')}, {l.get('state','')}" for l in locations[:3]
            )
            if len(locations) > 3:
                loc_str += f" +{len(locations)-3} more"
        else:
            loc_str = ""

        # Specialty chips HTML
        chips = "".join(f"<span class='spec-chip'>{s}</span>" for s in specialties[:6])

        # Accepting badge
        accepting_badge = (
            "<span style='color:#2E7D32;font-size:.8rem;'>✅ Accepting patients</span>"
            if accepting else
            "<span style='color:#B71C1C;font-size:.8rem;'>⏳ Call to confirm availability</span>"
        )

        html = f"""
        <div class='therapist-card'>
            <div class='therapist-name'>{name}</div>
            <div class='therapist-title'>{ptype} {accepting_badge}</div>
            <div style='margin:6px 0;font-size:.85rem;color:#555;'>{description[:200]}{'...' if len(description)>200 else ''}</div>
            <div>{chips}</div>
            <div class='contact-row'>
                {"📍 " + loc_str + "<br>" if loc_str else ""}
                {"📞 " + phone + "<br>" if phone else ""}
                {"🎯 Ages served: " + ages + "<br>" if ages else ""}
                {"💻 Sessions: " + ", ".join(session_types) + "<br>" if session_types else ""}
                {"🌐 Languages: " + ", ".join(languages) if languages else ""}
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

        # Individual providers within a clinic
        if providers_list:
            with st.expander(f"👥 {len(providers_list)} Providers at {name}"):
                for prov in providers_list:
                    prov_html = f"""
                    <div class='provider-sub'>
                        <strong>{prov['name']}</strong><br>
                        <span style='color:#666;'>{prov['title']}</span>
                        {"<br><a href='"+prov['profile_url']+"' target='_blank' style='font-size:.8rem;'>View profile →</a>" if prov.get('profile_url') else ""}
                    </div>
                    """
                    st.markdown(prov_html, unsafe_allow_html=True)

        # Action buttons
        btn_cols = st.columns(3)
        if website:
            with btn_cols[0]:
                st.link_button("🌐 Visit Website", website, use_container_width=True)
        if contact_form and contact_form != website:
            with btn_cols[1]:
                st.link_button("📅 Book / Contact", contact_form, use_container_width=True)
        if phone:
            with btn_cols[2]:
                st.markdown(
                    f"<div style='padding-top:8px;font-size:.9rem;'>📞 <strong>{phone}</strong></div>",
                    unsafe_allow_html=True,
                )
        st.markdown("---")


AGENT_META = {
    "retrieval":      ("badge-retrieval",      "📚 Knowledge Base (RAG)"),
    "recommendation": ("badge-recommendation", "💡 Recommendation Agent"),
    "direct":         ("badge-direct",         "💬 Direct Answer"),
    "crisis":         ("badge-crisis",         "🆘 Crisis Support"),
    "referral":       ("badge-referral",       "🩺 Provider Directory"),
}

for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # Crisis messages get a warning container
        if message["role"] == "assistant" and message.get("agent_used") == "crisis":
            st.error(message["content"])
        else:
            st.markdown(message["content"])

        if message["role"] == "assistant":
            agent_used = message.get("agent_used", "")
            badge_cls, badge_label = AGENT_META.get(agent_used, ("badge-direct", agent_used))

            st.markdown(
                f"<span class='agent-badge {badge_cls}'>{badge_label}</span>",
                unsafe_allow_html=True,
            )

            # --- Therapist referral cards ---
            referred_providers = message.get("referred_providers", [])
            if referred_providers:
                render_therapist_cards(referred_providers)

            # --- Source citations ---
            retrieved_docs = message.get("retrieved_docs", [])
            if retrieved_docs:
                with st.expander(f"📄 Sources ({len(retrieved_docs)} documents retrieved)"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        content_preview = doc["content"][:300].replace("\n", " ")
                        metadata = doc.get("metadata", {})
                        meta_str = ""
                        if metadata:
                            relevant = {k: v for k, v in metadata.items() if k in ["source", "category", "specialty", "topic"]}
                            if relevant:
                                meta_str = " · ".join(f"{k}: {v}" for k, v in relevant.items())

                        st.markdown(
                            f"""<div class='source-card'>
                            <strong>Source {i}</strong>
                            {"<br><small style='color:#888'>"+meta_str+"</small>" if meta_str else ""}
                            <br>{content_preview}{'...' if len(doc['content']) > 300 else ''}
                            </div>""",
                            unsafe_allow_html=True,
                        )

            # --- Feedback buttons ---
            trace_id = message.get("trace_id")
            if trace_id:
                feedback_given = message.get("feedback")
                if feedback_given is None:
                    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 10])
                    with fb_col1:
                        if st.button("👍", key=f"up_{msg_idx}", help="Helpful"):
                            ok = st.session_state.agent.submit_feedback(trace_id, 1.0, "Helpful")
                            st.session_state.messages[msg_idx]["feedback"] = "positive"
                            st.session_state.stats["feedback_pos"] += 1
                            st.rerun()
                    with fb_col2:
                        if st.button("👎", key=f"down_{msg_idx}", help="Not helpful"):
                            ok = st.session_state.agent.submit_feedback(trace_id, 0.0, "Not helpful")
                            st.session_state.messages[msg_idx]["feedback"] = "negative"
                            st.session_state.stats["feedback_neg"] += 1
                            st.rerun()
                else:
                    icon = "👍" if feedback_given == "positive" else "👎"
                    st.caption(f"{icon} Feedback submitted · thank you!")

# ------------------------------------------------------------------
# Chat input
# ------------------------------------------------------------------

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

prompt = st.chat_input("Ask your question here... (English or Bahasa Indonesia)")
if not prompt and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    if not st.session_state.get("agent_ready"):
        st.error("⚠️ Agent not initialized. Please check your configuration.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.agent.query(prompt)
                    agent_used = response.get("agent_used", "direct")
                    retrieved_docs = response.get("retrieved_docs", [])
                    referred_providers = response.get("referred_providers", [])
                    trace_id = response.get("trace_id")
                    detected_lang = response.get("detected_language", "english")

                    # Display
                    if agent_used == "crisis":
                        st.error(response["answer"])
                    else:
                        st.markdown(response["answer"])

                    badge_cls, badge_label = AGENT_META.get(agent_used, ("badge-direct", agent_used))
                    st.markdown(
                        f"<span class='agent-badge {badge_cls}'>{badge_label}</span>",
                        unsafe_allow_html=True,
                    )

                    # Therapist referral cards
                    if referred_providers:
                        render_therapist_cards(referred_providers)

                    # Citations
                    if retrieved_docs:
                        with st.expander(f"📄 Sources ({len(retrieved_docs)} documents retrieved)"):
                            for i, doc in enumerate(retrieved_docs, 1):
                                content_preview = doc["content"][:300].replace("\n", " ")
                                metadata = doc.get("metadata", {})
                                meta_str = ""
                                if metadata:
                                    relevant = {k: v for k, v in metadata.items() if k in ["source", "category", "specialty", "topic"]}
                                    if relevant:
                                        meta_str = " · ".join(f"{k}: {v}" for k, v in relevant.items())
                                st.markdown(
                                    f"""<div class='source-card'>
                                    <strong>Source {i}</strong>
                                    {"<br><small style='color:#888'>"+meta_str+"</small>" if meta_str else ""}
                                    <br>{content_preview}{'...' if len(doc['content']) > 300 else ''}
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

                    # Inline feedback buttons (newly answered message)
                    if trace_id:
                        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 10])
                        with fb_col1:
                            st.button("👍", key=f"up_new", help="Helpful")
                        with fb_col2:
                            st.button("👎", key=f"down_new", help="Not helpful")

                    # Update stats
                    stats = st.session_state.stats
                    stats["total"] += 1
                    if agent_used in stats:
                        stats[agent_used] += 1
                    if detected_lang == "indonesian":
                        stats["lang_id"] += 1
                    else:
                        stats["lang_en"] += 1

                    # Store message
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response["answer"],
                            "agent_used": agent_used,
                            "retrieved_docs": retrieved_docs,
                            "referred_providers": referred_providers,
                            "trace_id": trace_id,
                            "feedback": None,
                        }
                    )

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------

st.markdown("---")
st.markdown(
    """
<div style='text-align:center; color:#999; font-size:.8rem;'>
💚 <strong>MindPulse</strong> — Mental Health Knowledge & Referral Platform<br>
Built with LangChain · LangGraph · Qdrant · Cohere · Langfuse · Streamlit · OpenAI GPT-4o-mini
</div>
""",
    unsafe_allow_html=True,
)
