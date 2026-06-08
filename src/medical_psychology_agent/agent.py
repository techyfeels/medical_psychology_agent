"""Supervisor Agent using LangGraph for medical psychology queries."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, StateGraph

from medical_psychology_agent.config import Config
from medical_psychology_agent.prompts import (
    CRISIS_SYSTEM_PROMPT,
    DIRECT_SYSTEM_PROMPT,
    RECOMMENDATION_SYSTEM_PROMPT,
    REFERRAL_SYSTEM_PROMPT,
    RETRIEVAL_SYSTEM_PROMPT,
    SUPERVISOR_PROMPT,
    get_prompt_from_langfuse,
)
from medical_psychology_agent.rag_tool import RAGTool
from medical_psychology_agent.therapist_tool import TherapistFinder
from medical_psychology_agent.translator import detect_language


class AgentState(TypedDict):
    messages: List
    input: str
    context: str
    agent_decision: str
    final_answer: str
    retrieved_docs: List[Dict]
    detected_language: str
    referred_providers: List[Dict]


class MedicalPsychologyAgent:
    """Multi-agent supervisor for medical psychology queries.

    Graph: supervisor → retrieval_agent | direct_answer_agent | crisis_agent | recommendation_agent
    """

    def __init__(
        self,
        use_reranker: bool = True,
        use_translation: bool = True,
        use_langfuse: bool = True,
    ):
        Config.validate()
        self.chat_history: List = []

        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=0.3,
        )

        self.rag_tool = RAGTool(use_reranker=use_reranker, use_translation=use_translation)
        self.therapist_finder = TherapistFinder()

        self.use_langfuse = bool(
            use_langfuse
            and Config.LANGFUSE_SECRET_KEY
            and Config.LANGFUSE_PUBLIC_KEY
            and Config.LANGFUSE_BASE_URL
        )

        self.langfuse: Optional[Langfuse] = None
        self.langfuse_handler: Optional[CallbackHandler] = None

        if self.use_langfuse:
            os.environ["LANGFUSE_SECRET_KEY"] = Config.LANGFUSE_SECRET_KEY
            os.environ["LANGFUSE_PUBLIC_KEY"] = Config.LANGFUSE_PUBLIC_KEY
            os.environ["LANGFUSE_HOST"] = Config.LANGFUSE_BASE_URL
            self.langfuse = Langfuse()
            self.langfuse_handler = CallbackHandler()
            print("✅ Langfuse tracing enabled")
        else:
            if use_langfuse:
                print("⚠️ Langfuse keys missing - tracing disabled")
            else:
                print("ℹ️ Langfuse disabled")

        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("retrieval_agent", self._retrieval_agent_node)
        workflow.add_node("direct_answer_agent", self._direct_answer_node)
        workflow.add_node("crisis_agent", self._crisis_node)
        workflow.add_node("recommendation_agent", self._recommendation_node)
        workflow.add_node("referral_agent", self._referral_agent_node)

        workflow.set_entry_point("supervisor")

        workflow.add_conditional_edges(
            "supervisor",
            self._route_query,
            {
                "retrieval": "retrieval_agent",
                "direct": "direct_answer_agent",
                "crisis": "crisis_agent",
                "recommendation": "recommendation_agent",
                "referral": "referral_agent",
            },
        )

        workflow.add_edge("retrieval_agent", END)
        workflow.add_edge("direct_answer_agent", END)
        workflow.add_edge("crisis_agent", END)
        workflow.add_edge("recommendation_agent", END)
        workflow.add_edge("referral_agent", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _invoke_llm(self, messages: List):
        if self.langfuse_handler:
            return self.llm.invoke(messages, config={"callbacks": [self.langfuse_handler]})
        return self.llm.invoke(messages)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _supervisor_node(self, state: AgentState) -> AgentState:
        """Classify the query and set agent_decision to one of 4 routes."""
        input_text = state["input"]
        text_lower = (input_text or "").lower()

        # Detect language for analytics
        state["detected_language"] = detect_language(input_text)

        # 1. Crisis detection — highest priority
        crisis_keywords = [
            "bunuh diri", "ingin mati", "mau mati", "tidak mau hidup",
            "menyakiti diri", "tidak ingin hidup", "lelah hidup",
            "suicide", "suicidal", "kill myself", "end my life",
            "self harm", "self-harm", "cutting myself", "hurt myself",
            "want to die", "better off dead", "no reason to live",
        ]
        if any(k in text_lower for k in crisis_keywords):
            state["agent_decision"] = "crisis"
            print("🆘 Supervisor: crisis (rule-based)")
            return state

        # 2. Referral detection — find therapist/psychiatrist
        # Approach: check if text contains a provider word + an action word
        provider_words = [
            "therapist", "psychiatrist", "psychologist", "counselor",
            "psikiater", "terapis", "psikolog", "dokter jiwa",
        ]
        action_words = [
            "find", "need", "looking", "contact", "get", "book", "appointment",
            "recommend", "refer", "where", "how to reach", "number", "phone",
            "cari", "butuh", "mau ketemu", "kontak", "nomor", "rekomendasi",
        ]
        referral_exact = [
            "cari psikiater", "cari terapis", "cari dokter jiwa",
            "rekomendasi psikiater", "rekomendasi terapis",
            "kontak psikiater", "kontak terapis", "nomor psikiater",
            "have contact", "do you have contact",
            "mental health professional near", "therapist near",
            "psychiatrist near", "psychologist near",
        ]
        has_provider = any(w in text_lower for w in provider_words)
        has_action = any(w in text_lower for w in action_words)
        has_exact = any(k in text_lower for k in referral_exact)

        if has_exact or (has_provider and has_action):
            state["agent_decision"] = "referral"
            print("📋 Supervisor: referral (rule-based)")
            return state

        # 3. Recommendation detection
        recommendation_keywords = [
            "apa yang harus", "saran", "rekomendasi", "langkah-langkah",
            "cara terbaik", "apa yang perlu", "tips untuk", "strategi untuk",
            "what should i", "recommend", "suggest", "give me advice",
            "steps to", "how can i improve", "what can i do about",
            "guide me", "help me with", "what are the best ways",
        ]
        if any(k in text_lower for k in recommendation_keywords):
            state["agent_decision"] = "recommendation"
            print("💡 Supervisor: recommendation (rule-based)")
            return state

        # 3. Medical RAG detection
        rag_keywords = [
            "depresi", "depression", "anxiety", "kecemasan", "insomnia",
            "bipolar", "therapy", "therapist", "cbt", "dbt", "ptsd",
            "panic", "schizo", "schizophrenia", "adhd", "ocd", "trauma",
            "disorder", "mental health", "gangguan", "penyakit jiwa",
            "gejala", "symptoms", "diagnosis", "treatment", "medication",
            "phobia", "eating disorder", "anoreksia", "bulimia",
            "narcissistic", "borderline", "autism", "dementia",
        ]
        if any(k in text_lower for k in rag_keywords):
            state["agent_decision"] = "retrieval"
            print("🧠 Supervisor: retrieval (rule-based)")
            return state

        # 4. LLM-based routing for ambiguous queries
        prompt_template = (
            get_prompt_from_langfuse("medical_psychology_supervisor")
            or SUPERVISOR_PROMPT
        )
        prompt = prompt_template.format(
            input=input_text,
            query=input_text,
            context="",
            documents="",
        )

        response = self._invoke_llm([SystemMessage(content=prompt)])
        decision_text = (response.content or "").lower()

        if "crisis" in decision_text:
            decision = "crisis"
        elif "referral" in decision_text:
            decision = "referral"
        elif "recommendation" in decision_text:
            decision = "recommendation"
        elif any(k in decision_text for k in ["retrieval", "search", "knowledge", "complex"]):
            decision = "retrieval"
        else:
            decision = "direct"

        state["agent_decision"] = decision
        print(f"🧠 Supervisor: {decision} (LLM-based)")
        return state

    def _route_query(self, state: AgentState) -> str:
        return state["agent_decision"]

    def _retrieval_agent_node(self, state: AgentState) -> AgentState:
        """Retrieve docs and answer using knowledge base context."""
        print("📚 Retrieval agent processing...")

        documents = self.rag_tool.retrieve(state["input"])
        context = self.rag_tool.format_context(documents)
        state["context"] = context
        state["retrieved_docs"] = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in documents
        ]

        system_msg = SystemMessage(content=RETRIEVAL_SYSTEM_PROMPT.format(context=context))
        response = self._invoke_llm([system_msg] + list(state["messages"]))

        state["final_answer"] = response.content
        state["messages"].append(AIMessage(content=response.content))
        return state

    def _direct_answer_node(self, state: AgentState) -> AgentState:
        """Answer greetings and small talk directly."""
        print("💬 Direct answer agent processing...")

        system_msg = SystemMessage(content=DIRECT_SYSTEM_PROMPT)
        response = self._invoke_llm([system_msg] + list(state["messages"]))

        state["final_answer"] = response.content
        state["messages"].append(AIMessage(content=response.content))
        return state

    def _crisis_node(self, state: AgentState) -> AgentState:
        """Respond with empathy + crisis resources."""
        print("🆘 Crisis agent processing...")

        system_msg = SystemMessage(content=CRISIS_SYSTEM_PROMPT)
        response = self._invoke_llm([system_msg] + list(state["messages"]))

        state["final_answer"] = response.content
        state["messages"].append(AIMessage(content=response.content))
        return state

    def _referral_agent_node(self, state: AgentState) -> AgentState:
        """Search therapist directory and suggest relevant providers."""
        print("📋 Referral agent processing...")

        input_text = state["input"]
        text_lower = input_text.lower()

        # Infer specialty from query
        specialty_hints = {
            "anxiety": "anxiety", "depresi": "depression", "depression": "depression",
            "ptsd": "ptsd", "trauma": "trauma", "adhd": "adhd", "ocd": "ocd",
            "bipolar": "bipolar", "insomnia": "insomnia", "addiction": "addiction",
            "couples": "couples", "children": "children", "adolescent": "adolescent",
        }
        detected_specialty = next(
            (v for k, v in specialty_hints.items() if k in text_lower), None
        )

        # Infer area from query
        area_hints = {
            "arlington": "arlington", "alexandria": "alexandria",
            "fairfax": "fairfax", "dc": "dc", "washington": "washington",
            "virginia": "virginia", "northern virginia": "northern virginia",
        }
        detected_area = next(
            (v for k, v in area_hints.items() if k in text_lower), None
        )

        providers = self.therapist_finder.search(
            specialty=detected_specialty, area=detected_area, limit=5
        )
        if not providers:
            providers = self.therapist_finder.get_all(limit=5)

        state["referred_providers"] = providers

        provider_context = self.therapist_finder.format_for_agent(providers)
        system_msg = SystemMessage(
            content=REFERRAL_SYSTEM_PROMPT.format(providers=provider_context)
        )
        response = self._invoke_llm([system_msg] + list(state["messages"]))

        state["final_answer"] = response.content
        state["messages"].append(AIMessage(content=response.content))
        return state

    def _recommendation_node(self, state: AgentState) -> AgentState:
        """Retrieve context and produce structured self-help recommendations."""
        print("💡 Recommendation agent processing...")

        documents = self.rag_tool.retrieve(state["input"])
        context = self.rag_tool.format_context(documents)
        state["context"] = context
        state["retrieved_docs"] = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in documents
        ]

        system_msg = SystemMessage(content=RECOMMENDATION_SYSTEM_PROMPT.format(context=context))
        response = self._invoke_llm([system_msg] + list(state["messages"]))

        state["final_answer"] = response.content
        state["messages"].append(AIMessage(content=response.content))
        return state

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def query(self, user_input: str) -> dict:
        print(f"\n{'='*60}")
        print(f"🔍 Processing: {user_input}")
        print(f"{'='*60}")

        # Create an explicit Langfuse trace per query for feedback scoring
        trace_id: Optional[str] = None
        if self.use_langfuse and self.langfuse:
            try:
                trace = self.langfuse.trace(
                    name="medical_psychology_query",
                    input={"query": user_input},
                    tags=["medical", "psychology"],
                )
                trace_id = trace.id
            except Exception as e:
                print(f"⚠️ Langfuse trace creation failed: {e}")

        history = self.chat_history[-6:]
        initial_state: AgentState = {
            "messages": history + [HumanMessage(content=user_input)],
            "input": user_input,
            "context": "",
            "agent_decision": "",
            "final_answer": "",
            "retrieved_docs": [],
            "detected_language": "english",
            "referred_providers": [],
        }

        final_state = self.graph.invoke(initial_state)

        self.chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=final_state["final_answer"]),
        ])

        print(f"\n✅ Answered via: {final_state['agent_decision']} agent")
        return {
            "answer": final_state["final_answer"],
            "agent_used": final_state["agent_decision"],
            "context_used": final_state["context"] if final_state["context"] else None,
            "retrieved_docs": final_state.get("retrieved_docs", []),
            "referred_providers": final_state.get("referred_providers", []),
            "detected_language": final_state.get("detected_language", "english"),
            "query": user_input,
            "trace_id": trace_id,
        }

    def submit_feedback(self, trace_id: str, value: float, comment: str = "") -> bool:
        """Submit user feedback to Langfuse. value: 1.0 = positive, 0.0 = negative."""
        if not (self.use_langfuse and self.langfuse and trace_id):
            return False
        try:
            self.langfuse.score(
                trace_id=trace_id,
                name="user_feedback",
                value=value,
                comment=comment,
            )
            return True
        except Exception as e:
            print(f"⚠️ Feedback submission failed: {e}")
            return False

    def chat(self):
        """Interactive CLI chat mode."""
        print("\n" + "="*60)
        print("🏥 Medical Psychology Assistant")
        print("="*60)
        print("Type 'quit' to exit.\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "bye", "keluar"]:
                print("\n👋 Goodbye! Take care of your mental health.")
                break
            if not user_input:
                continue
            response = self.query(user_input)
            print(f"\nAssistant: {response['answer']}\n")


if __name__ == "__main__":
    agent = MedicalPsychologyAgent(use_reranker=True, use_translation=True, use_langfuse=True)

    test_queries = [
        "Hello!",
        "What are the symptoms of anxiety disorder?",
        "Apa itu gangguan depresi?",
        "What should I do to manage stress better?",
    ]

    for q in test_queries:
        r = agent.query(q)
        print(f"\n📝 Answer: {r['answer'][:200]}...")
        print(f"🔧 Agent: {r['agent_used']}")
        print("="*60 + "\n")
