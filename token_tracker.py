"""
Tracks token usage across agent steps and estimates cost.
Uses Groq pricing for Llama 3.3 70B.
"""

import tiktoken
from langchain_core.callbacks import BaseCallbackHandler


# groq pricing per 1M tokens (as of 2025)
PRICE_INPUT_PER_M = 0.59    # $/1M input tokens
PRICE_OUTPUT_PER_M = 0.79   # $/1M output tokens


def count_tokens(text, model="cl100k_base"):
    """Rough token count using tiktoken. Works well enough for estimates."""
    try:
        enc = tiktoken.get_encoding(model)
        return len(enc.encode(text))
    except Exception:
        # fallback: ~4 chars per token
        return len(text) // 4


class TokenTracker(BaseCallbackHandler):
    """
    LangChain callback that records token usage from each LLM call.
    Accumulates totals so we can show a summary at the end.
    """

    def __init__(self):
        self.steps = []
        self.total_input = 0
        self.total_output = 0

    def on_llm_end(self, response, **kwargs):
        """Called after each LLM generation. Pull token counts from metadata."""
        for gen_list in response.generations:
            for gen in gen_list:
                usage = {}
                # groq puts usage in generation_info or message.usage_metadata
                if hasattr(gen, "generation_info") and gen.generation_info:
                    usage = gen.generation_info.get("usage", {})
                elif hasattr(gen, "message") and hasattr(gen.message, "usage_metadata"):
                    meta = gen.message.usage_metadata
                    if meta:
                        usage = {
                            "prompt_tokens": meta.get("input_tokens", 0),
                            "completion_tokens": meta.get("output_tokens", 0),
                            "total_tokens": meta.get("total_tokens", 0),
                        }

                input_tk = usage.get("prompt_tokens", 0)
                output_tk = usage.get("completion_tokens", 0)

                self.total_input += input_tk
                self.total_output += output_tk
                self.steps.append({
                    "step": len(self.steps) + 1,
                    "input_tokens": input_tk,
                    "output_tokens": output_tk,
                })

    @property
    def total_tokens(self):
        return self.total_input + self.total_output

    @property
    def estimated_cost(self):
        """Cost in USD based on Groq pricing."""
        input_cost = (self.total_input / 1_000_000) * PRICE_INPUT_PER_M
        output_cost = (self.total_output / 1_000_000) * PRICE_OUTPUT_PER_M
        return input_cost + output_cost

    def get_summary(self):
        return {
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost, 6),
            "num_llm_calls": len(self.steps),
            "per_step": self.steps,
        }
