"""
LLM config — picks up Groq if the API key is available,
otherwise tries Ollama as a fallback.
"""

import os
from dotenv import load_dotenv

load_dotenv()

_cached_llm = None
_cached_name = None


def get_llm():
    """Return a configured LLM instance. Caches after first call."""
    global _cached_llm, _cached_name

    if _cached_llm is not None:
        return _cached_llm, _cached_name

    # try groq first (cloud, fast)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=0,
            )
            name = "Groq Cloud (llama-3.3-70b-versatile)"
            print(f"[OK] LLM Active: {name}")
            _cached_llm, _cached_name = llm, name
            return llm, name
        except Exception as e:
            print(f"[WARN] Groq failed: {e}, trying Ollama...")

    # fallback to local ollama
    print("[INFO] No GROQ_API_KEY, trying Ollama...")
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3.2:1b", temperature=0)
        name = "Ollama Local (llama3.2:1b)"
        print(f"[OK] LLM Active: {name}")
        _cached_llm, _cached_name = llm, name
        return llm, name
    except Exception as e:
        print(f"[ERROR] Ollama failed too: {e}")

    raise RuntimeError(
        "No LLM available — set GROQ_API_KEY or install Ollama"
    )


if __name__ == "__main__":
    llm, name = get_llm()
    print(f"Ready: {name}")
