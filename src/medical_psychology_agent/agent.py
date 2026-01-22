"""Supervisor Agent using LangGraph for medical psychology queries."""

from __future__ import annotations

import os
from typing import List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, StateGraph

from medical_psychology_agent.config import Config
from medical_psychology_agent.prompts import (
    DIRECT_ANSWER_PROMPT,
    RETRIEVAL_AGENT_PROMPT,
    SUPERVISOR_PROMPT,
    get_prompt_from_langfuse,
)
from medical_psychology_agent.rag_tool import RAGTool


class AgentState(TypedDict):
    """State for the agent graph"""

    messages: List[HumanMessage | AIMessage | SystemMessage]
    input: str
    context: str
    agent_decision: str
    final_answer: str


class MedicalPsychologyAgent:
    """Supervisor agent for medical psychology queries"""

    def __init__(
        self,
        use_reranker: bool = True,
        use_translation: bool = True,
        use_langfuse: bool = True,
    ):
        """Initialize the supervisor agent.

        Args:
            use_reranker: Enable Cohere reranker for RAG
            use_translation: Enable query translation for Indonesian queries
            use_langfuse: Enable Langfuse tracing
        """

        Config.validate()
        self.chat_history = []

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=0.3,
        )

        # Initialize RAG tool with translation support
        self.rag_tool = RAGTool(use_reranker=use_reranker, use_translation=use_translation)

        # Initialize Langfuse (SAFE VERSION)
        self.use_langfuse = bool(
            use_langfuse
            and Config.LANGFUSE_SECRET_KEY
            and Config.LANGFUSE_PUBLIC_KEY
            and Config.LANGFUSE_BASE_URL
        )

        if self.use_langfuse:
            # Langfuse SDK reads credentials from env vars
            os.environ["LANGFUSE_SECRET_KEY"] = Config.LANGFUSE_SECRET_KEY
            os.environ["LANGFUSE_PUBLIC_KEY"] = Config.LANGFUSE_PUBLIC_KEY
            os.environ["LANGFUSE_HOST"] = Config.LANGFUSE_BASE_URL

            # Optional client (useful for prompt management)
            self.langfuse = Langfuse()

            # LangChain callback handler (tracing)
            self.langfuse_handler = CallbackHandler()

            print("✅ Langfuse tracing enabled")
        else:
            self.langfuse = None
            self.langfuse_handler = None
            if use_langfuse:
                print("⚠️ Langfuse keys missing - tracing disabled")
            else:
                print("ℹ️ Langfuse disabled")

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph supervisor workflow"""

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("retrieval_agent", self._retrieval_agent_node)
        workflow.add_node("direct_answer_agent", self._direct_answer_node)

        # Set entry point
        workflow.set_entry_point("supervisor")

        # Add conditional edges from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self._route_query,
            {
                "retrieval": "retrieval_agent",
                "direct": "direct_answer_agent",
            },
        )

        # Add edges to END
        workflow.add_edge("retrieval_agent", END)
        workflow.add_edge("direct_answer_agent", END)

        return workflow.compile()

    def _supervisor_node(self, state: AgentState) -> AgentState:
        """Supervisor decides which agent to use"""

        input_text = state["input"]
        text_lower = (input_text or "").lower()

        # 1) QUICK RULE-BASED ROUTING (guarantee RAG for doc-related queries)
        rag_keywords = [
            "depresi", "depression", "anxiety", "kecemasan", "insomnia",
            "bipolar", "therapy", "therapist", "cbt", "ptsd",
            "panic", "suic", "suicide", "schizo", "schizophrenia",
            "adhd", "ocd", "trauma"
        ]

        if any(k in text_lower for k in rag_keywords):
            state["agent_decision"] = "retrieval"
            print("🧠 Supervisor decision: retrieval (rule-based)")
            return state

        # 2) Otherwise: use LLM supervisor (Langfuse → fallback local)
        from medical_psychology_agent.prompts import get_prompt_from_langfuse

        prompt_template = (
            get_prompt_from_langfuse("medical_psychology_supervisor")
            or SUPERVISOR_PROMPT
        )

        # Safety guard in case {context} exists
        prompt = prompt_template.format(
            input=input_text,
            query=input_text,
            context="",
            documents=""
        )

        messages = [SystemMessage(content=prompt)]

        if self.langfuse_handler:
            response = self.llm.invoke(
                messages,
                config={"callbacks": [self.langfuse_handler]}
            )
        else:
            response = self.llm.invoke(messages)

        decision_text = (response.content or "").lower()

        if any(k in decision_text for k in ["retrieval", "search", "knowledge base", "complex"]):
            decision = "retrieval"
        else:
            decision = "direct"

        state["agent_decision"] = decision
        print(f"🧠 Supervisor decision: {decision}")

        return state


    def _route_query(self, state: AgentState) -> str:
        """Route to appropriate agent based on supervisor decision"""
        return state["agent_decision"]

    def _retrieval_agent_node(self, state: AgentState) -> AgentState:
        """Retrieval agent with RAG and translation support"""

        input_text = state["input"]
        print("📚 Retrieval agent processing query...")

        # Retrieve context (translation happens inside RAGTool)
        documents = self.rag_tool.retrieve(input_text)
        context = self.rag_tool.format_context(documents)
        state["context"] = context

        prompt = RETRIEVAL_AGENT_PROMPT.format(context=context, input=input_text)
        messages = [SystemMessage(content=prompt)]

        if self.langfuse_handler:
            response = self.llm.invoke(messages, config={"callbacks": [self.langfuse_handler]})
        else:
            response = self.llm.invoke(messages)

        state["final_answer"] = response.content
        state["messages"].append(AIMessage(content=response.content))
        return state

    def _direct_answer_node(self, state: AgentState) -> AgentState:
        """Direct answer agent for simple queries"""

        input_text = state["input"]
        print("💬 Direct answer agent processing query...")

        prompt = DIRECT_ANSWER_PROMPT.format(input=input_text)
        messages = [SystemMessage(content=prompt)]

        if self.langfuse_handler:
            response = self.llm.invoke(messages, config={"callbacks": [self.langfuse_handler]})
        else:
            response = self.llm.invoke(messages)

        state["final_answer"] = response.content
        state["messages"].append(AIMessage(content=response.content))
        return state

    def query(self, user_input: str) -> dict:
        print(f"\n{'='*60}")
        print(f"🔍 Processing query: {user_input}")
        print(f"{'='*60}")

        # === ambil 3 percakapan terakhir ===
        history = self.chat_history[-6:]  # 3 user + 3 assistant

        initial_state = {
        "messages": history + [HumanMessage(content=user_input)],
        "input": user_input,
        "context": "",
        "agent_decision": "",
        "final_answer": ""}

        final_state = self.graph.invoke(initial_state)

        # === simpan history ===
        self.chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=final_state["final_answer"])])

        response = {
        "answer": final_state["final_answer"],
        "agent_used": final_state["agent_decision"],
        "context_used": final_state["context"] if final_state["context"] else None,
        "query": user_input}

        print(f"\n✅ Response generated using: {response['agent_used']} agent")
        return response


    def chat(self):
        """Interactive chat mode"""

        print("\n" + "=" * 60)
        print("🏥 Medical Psychology Assistant")
        print("=" * 60)
        print("Ask me anything about medical psychology!")
        print("Tanya apa saja tentang psikologi medis!")
        print("Type 'quit' or 'exit' to end the conversation.\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "bye", "keluar"]:
                print("\n👋 Goodbye! Take care of your mental health!")
                print("👋 Sampai jumpa! Jaga kesehatan mental Anda!")
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
        "Bagaimana cara mengatasi insomnia?",
    ]

    for q in test_queries:
        r = agent.query(q)
        print(f"\n📝 Answer: {r['answer'][:200]}...")
        print(f"🔧 Agent: {r['agent_used']}")
        print("=" * 60 + "\n")
