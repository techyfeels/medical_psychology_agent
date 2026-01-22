import string
from langfuse import Langfuse

from medical_psychology_agent.config import Config


def _placeholders(t: str) -> list[str]:
    fmt = string.Formatter()
    names = []
    for _, name, _, _ in fmt.parse(t):
        if name:
            names.append(name)
    return sorted(set(names))


def _looks_like_code(text: str) -> bool:
    bad_markers = [
        "SUPERVISOR_PROMPT =",
        "RETRIEVAL_AGENT_PROMPT =",
        "DIRECT_ANSWER_PROMPT =",
        "LANGFUSE_PROMPTS",
        "EXAMPLE_QUERIES",
        "def ",
        "class ",
        "{\n",  # sering muncul kalau yang ketarik JSON/dict
    ]
    return any(m in text for m in bad_markers)


def _is_valid_prompt(prompt_name: str, text: str) -> tuple[bool, list[str]]:
    """
    Validasi supaya prompt Langfuse yang ketarik itu bener-bener template prompt (bukan kode/dict).
    Rules:
      - supervisor: WAJIB punya {input}. Boleh hanya {input}.
      - retrieval: WAJIB punya {input} dan {context}.
      - direct: WAJIB punya {input}.
    """
    if not isinstance(text, str) or len(text.strip()) < 20:
        return False, []

    if _looks_like_code(text):
        return False, _placeholders(text)

    ph = set(_placeholders(text))

    if prompt_name == "medical_psychology_supervisor":
        ok = ph == {"input"} or ph.issubset({"input"})  # strict: idealnya cuma {input}
        return ok, sorted(ph)

    if prompt_name == "medical_psychology_retrieval":
        ok = {"input", "context"}.issubset(ph)
        return ok, sorted(ph)

    if prompt_name == "medical_psychology_direct":
        ok = ph == {"input"} or ph.issubset({"input"})
        return ok, sorted(ph)

    # unknown prompt name -> reject
    return False, sorted(ph)


def get_prompt_from_langfuse(prompt_name: str, label: str = "production") -> str:
    """
    Fetch prompt from Langfuse (prefer label=production).
    Fallback ke prompt lokal kalau:
      - prompt tidak ada (404)
      - isinya bukan template prompt (kode/python/json)
      - placeholder tidak sesuai
    """
    try:
        # Langfuse will read credentials from env vars:
        # LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST
        langfuse = Langfuse()

        try:
            p = langfuse.get_prompt(prompt_name, label=label)
        except Exception:
            # fallback: coba tanpa label (kadang env/SDK beda behaviour)
            p = langfuse.get_prompt(prompt_name)

        # Ambil raw template string
        template = None
        if isinstance(p, str):
            template = p
        else:
            template = (
                getattr(p, "prompt", None)
                or getattr(p, "text", None)
                or getattr(p, "content", None)
            )

        ok, ph = _is_valid_prompt(prompt_name, template or "")
        if not ok:
            raise ValueError(
                f"Invalid Langfuse prompt content for {prompt_name}. Placeholders={ph}"
            )

        return template

    except Exception as e:
        print(f"⚠️ Langfuse prompt fetch failed ({prompt_name}): {e}")

        # fallback to local
        if prompt_name == "medical_psychology_supervisor":
            return SUPERVISOR_PROMPT
        if prompt_name == "medical_psychology_retrieval":
            return RETRIEVAL_AGENT_PROMPT
        return DIRECT_ANSWER_PROMPT


# =========================
# LOCAL FALLBACK PROMPTS
# =========================

SUPERVISOR_PROMPT = """You are a Medical Psychology Supervisor Agent coordinating a team of specialized assistants.

Available Assistants:
- retrieval_agent: Searches the medical psychology knowledge base.
- direct_answer_agent: Answers simple/general questions without retrieval.

Rules:
- If the user asks about a mental health concept, symptom, disorder, therapy, treatment, diagnosis, medication, or wants an explanation → choose retrieval_agent.
- If it is a greeting/small talk → choose direct_answer_agent.

User query: {input}

Return ONLY one word: retrieval or direct
"""

RETRIEVAL_AGENT_PROMPT = """You are a Medical Psychology Information Retrieval Specialist.

LANGUAGE HANDLING:
- Detect the user's language (Indonesian or English)
- Respond in the SAME language as the user
- The knowledge base is in English - translate information naturally

Rules:
1) Use the provided context to answer accurately.
2) If context is insufficient, say so clearly and answer cautiously.
3) Be empathetic and add a brief safety disclaimer when relevant.

Context:
{context}

User Query: {input}
"""

DIRECT_ANSWER_PROMPT = """You are a friendly Medical Psychology Assistant.

LANGUAGE HANDLING:
- Detect the user's language (Indonesian or English)
- Respond in the SAME language as the user

Rules:
- For greetings/small talk: respond warmly.
- Do not invent citations or claim you retrieved documents.

User Query: {input}
"""
