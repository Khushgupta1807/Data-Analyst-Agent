"""
LLM config — supports multiple providers.
Pass in a provider name and API key, get back a LangChain LLM.
"""

PROVIDERS = {
    "Groq": {
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "default": "llama-3.3-70b-versatile",
        "key_prefix": "gsk_",
        "env_var": "GROQ_API_KEY",
    },
    "OpenAI": {
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "default": "gpt-4o-mini",
        "key_prefix": "sk-",
        "env_var": "OPENAI_API_KEY",
    },
    "Google Gemini": {
        "models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        "default": "gemini-2.0-flash",
        "key_prefix": "AI",
        "env_var": "GOOGLE_API_KEY",
    },
    "Anthropic": {
        "models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
        "default": "claude-sonnet-4-20250514",
        "key_prefix": "sk-ant-",
        "env_var": "ANTHROPIC_API_KEY",
    },
}


def get_llm(provider, api_key, model=None):
    """
    Create a LangChain chat model for the given provider.
    Returns (llm, display_name) tuple.
    """
    if not api_key:
        raise ValueError("API key is required")

    config = PROVIDERS.get(provider)
    if not config:
        raise ValueError(f"Unknown provider: {provider}")

    model_name = model or config["default"]

    if provider == "Groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model_name, api_key=api_key, temperature=0)

    elif provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)

    elif provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)

    elif provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model_name, api_key=api_key, temperature=0)

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    display_name = f"{provider} ({model_name})"
    print(f"[OK] LLM Active: {display_name}")
    return llm, display_name


def detect_provider(api_key):
    """Try to guess the provider from the key prefix."""
    if not api_key:
        return None
    for name, config in PROVIDERS.items():
        if api_key.startswith(config["key_prefix"]):
            return name
    return None
