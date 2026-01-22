import string
from medical_psychology_agent.prompts import (
    get_prompt_from_langfuse,
    SUPERVISOR_PROMPT,
    RETRIEVAL_AGENT_PROMPT,
    DIRECT_ANSWER_PROMPT,
)

def placeholders_from_template(t: str):
    fmt = string.Formatter()
    names = []
    for _, name, _, _ in fmt.parse(t):
        if name:
            names.append(name)
    return sorted(set(names))

def inspect_prompt(name: str, template: str):
    ph = placeholders_from_template(template)
    print("\n---", name, "---")
    print("placeholders:", ph)
    print("preview:", template[:200].replace("\n", "\\n"))

def main():
    # 1️⃣ Supervisor
    try:
        sup = get_prompt_from_langfuse("medical_psychology_supervisor")
        inspect_prompt("LANGFUSE supervisor", sup)
    except Exception as e:
        print("❌ Langfuse supervisor failed:", e)
        inspect_prompt("LOCAL supervisor", SUPERVISOR_PROMPT)

    # 2️⃣ Retrieval
    try:
        ret = get_prompt_from_langfuse("medical_psychology_retrieval")
        inspect_prompt("LANGFUSE retrieval", ret)
    except Exception as e:
        print("❌ Langfuse retrieval failed:", e)
        inspect_prompt("LOCAL retrieval", RETRIEVAL_AGENT_PROMPT)

    # 3️⃣ Direct
    try:
        direc = get_prompt_from_langfuse("medical_psychology_direct")
        inspect_prompt("LANGFUSE direct", direc)
    except Exception as e:
        print("❌ Langfuse direct failed:", e)
        inspect_prompt("LOCAL direct", DIRECT_ANSWER_PROMPT)

if __name__ == "__main__":
    main()
