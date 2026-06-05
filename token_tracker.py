"""
Tracks token usage across agent steps and estimates cost.
Supports Groq, OpenAI, Gemini, and Anthropic.
"""

import tiktoken
from langchain_core.callbacks import BaseCallbackHandler


# pricing per 1M tokens (approximate, mid-2025)
PRICING = {
    "Groq": {"input": 0.59, "output": 0.79},
    "OpenAI": {"input": 0.15, "output": 0.60},       # gpt-4o-mini
    "Google Gemini": {"input": 0.075, "output": 0.30}, # gemini-2.0-flash
    "Anthropic": {"input": 3.00, "output": 15.00},     # claude-sonnet
}


def count_tokens(text, model="cl100k_base"):
    """Rough token count using tiktoken. Good enough for estimates."""
    try:
        enc = tiktoken.get_encoding(model)
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


class TokenTracker(BaseCallbackHandler):
    """
    LangChain callback that grabs token usage after each LLM call.
    Works across providers by checking multiple metadata locations.
    """

    def __init__(self, provider="Groq"):
        self.provider = provider
        self.steps = []
        self.total_input = 0
        self.total_output = 0

    def on_llm_end(self, response, **kwargs):
        """Pull token counts from wherever the provider puts them."""
        input_tk = 0
        output_tk = 0

        # method 1: llm_output (OpenAI, Groq often use this)
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            if token_usage:
                input_tk = token_usage.get("prompt_tokens", 0)
                output_tk = token_usage.get("completion_tokens", 0)

        # method 2: check each generation's metadata
        if input_tk == 0 and output_tk == 0:
            for gen_list in response.generations:
                for gen in gen_list:
                    # check generation_info
                    if hasattr(gen, "generation_info") and gen.generation_info:
                        info = gen.generation_info
                        # some providers put it directly
                        if "usage" in info:
                            u = info["usage"]
                            input_tk = u.get("prompt_tokens", u.get("input_tokens", 0))
                            output_tk = u.get("completion_tokens", u.get("output_tokens", 0))
                        # anthropic style
                        elif "usage" not in info and "input_tokens" in info:
                            input_tk = info.get("input_tokens", 0)
                            output_tk = info.get("output_tokens", 0)

                    # check message.usage_metadata (newer langchain)
                    if input_tk == 0 and hasattr(gen, "message"):
                        msg = gen.message
                        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                            meta = msg.usage_metadata
                            input_tk = meta.get("input_tokens", 0)
                            output_tk = meta.get("output_tokens", 0)

                        # also check response_metadata
                        if input_tk == 0 and hasattr(msg, "response_metadata") and msg.response_metadata:
                            rm = msg.response_metadata
                            if "token_usage" in rm:
                                tu = rm["token_usage"]
                                input_tk = tu.get("prompt_tokens", 0)
                                output_tk = tu.get("completion_tokens", 0)
                            elif "usage" in rm:
                                tu = rm["usage"]
                                input_tk = tu.get("prompt_tokens", tu.get("input_tokens", 0))
                                output_tk = tu.get("completion_tokens", tu.get("output_tokens", 0))

        self.total_input += input_tk
        self.total_output += output_tk
        self.steps.append({
            "step": len(self.steps) + 1,
            "input_tokens": input_tk,
            "output_tokens": output_tk,
            "total": input_tk + output_tk,
        })

    @property
    def total_tokens(self):
        return self.total_input + self.total_output

    @property
    def estimated_cost(self):
        pricing = PRICING.get(self.provider, PRICING["Groq"])
        input_cost = (self.total_input / 1_000_000) * pricing["input"]
        output_cost = (self.total_output / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def get_summary(self):
        return {
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost, 6),
            "num_llm_calls": len(self.steps),
            "per_step": self.steps,
            "provider": self.provider,
        }
