"""
LLM Configuration Module
Auto-detects and configures the best available LLM:
  1. Groq Cloud (if GROQ_API_KEY is set)
  2. Ollama Local (if Ollama is running)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Cache to avoid re-initializing on every call
_cached_llm = None
_cached_name = None


def get_llm():
    """
    Returns a configured LangChain LLM instance.
    Priority: Groq Cloud -> Ollama Local.
    Results are cached after first successful call.
    """
    global _cached_llm, _cached_name

    if _cached_llm is not None:
        return _cached_llm, _cached_name

    # --- Option 1: Groq Cloud ---
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=groq_api_key,
                temperature=0,
            )
            llm_name = "Groq Cloud (llama-3.3-70b-versatile)"
            print(f"[OK] LLM Active: {llm_name}")
            _cached_llm, _cached_name = llm, llm_name
            return llm, llm_name
        except Exception as e:
            print(f"[WARN] Groq init failed: {e}. Falling back to Ollama...")

    # --- Option 2: Ollama Local ---
    print("[INFO] GROQ_API_KEY not found. Trying Ollama local...")

    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3.2:1b", temperature=0)
        llm_name = "Ollama Local (llama3.2:1b)"
        print(f"[OK] LLM Active: {llm_name}")
        _cached_llm, _cached_name = llm, llm_name
        return llm, llm_name
    except Exception as e:
        print(f"[ERROR] Ollama LLM init failed: {e}")

    # --- No LLM available ---
    raise RuntimeError(
        "No LLM backend available. Set GROQ_API_KEY or install Ollama."
    )


# Run on import so the active LLM is printed at startup
if __name__ == "__main__":
    llm, name = get_llm()
    print(f"\nReady to use: {name}")
