"""
Main agent logic — builds a ReAct agent that can write + run Python code
to analyse a CSV, create plotly charts, and produce an insight report.

Includes prompt compression to cut token usage on large datasets.
"""

import io
import os
import pandas as pd
from langchain_experimental.tools import PythonREPLTool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from llm_config import get_llm
from token_tracker import TokenTracker, count_tokens


# --- prompt compression ---

MAX_HEAD_ROWS = 3          # show 3 rows instead of 5
MAX_COLS_IN_DESCRIBE = 12  # cap describe output for wide datasets
MAX_STR_COL_WIDTH = 30     # truncate long string values
MAX_INFO_LINES = 25        # cap df.info() output

def compress_head(df):
    """Fewer rows, truncated strings, saves ~40% tokens vs default."""
    sample = df.head(MAX_HEAD_ROWS).copy()
    for col in sample.select_dtypes(include=["object"]):
        sample[col] = sample[col].astype(str).str[:MAX_STR_COL_WIDTH]
    return sample.to_string()

def compress_info(df):
    """Compact version of df.info() — just dtypes and null counts."""
    buf = io.StringIO()
    df.info(buf=buf)
    lines = buf.getvalue().split("\n")
    # keep header + column lines + footer, skip the separator
    if len(lines) > MAX_INFO_LINES:
        kept = lines[:5] + lines[5:MAX_INFO_LINES-2] + ["...", lines[-2]]
        return "\n".join(kept)
    return "\n".join(lines)

def compress_describe(df):
    """If too many columns, only describe the most interesting ones."""
    if df.shape[1] > MAX_COLS_IN_DESCRIBE:
        # pick numeric cols first, then fill with categoricals
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        keep = (num_cols + cat_cols)[:MAX_COLS_IN_DESCRIBE]
        desc = df[keep].describe(include="all").to_string()
        return f"(showing {len(keep)}/{df.shape[1]} columns)\n{desc}"
    return df.describe(include="all").to_string()


def build_prompt(csv_path, df_head, df_info, df_describe):
    """Build the full ReAct prompt with compressed dataset context."""

    instructions = f"""You are an expert data analyst. You have access to a Python REPL tool.
Your job is to analyse a CSV dataset and produce visualisations and insights.

DATASET LOCATION: {csv_path}
The file is already on disk — load it with pandas.

Here is a preview of the dataset:

--- HEAD ---
{df_head}

--- INFO ---
{df_info}

--- DESCRIBE ---
{df_describe}

YOUR TASK (follow these steps in order):

1. Load the CSV using `pd.read_csv('{csv_path}')`.

2. Clean the data: handle missing values (drop or fill as appropriate).

3. Create exactly 3 Plotly Express charts showing the most interesting patterns:
   - Chart 1: distribution/histogram
   - Chart 2: relationship/correlation (scatter, bar, etc.)
   - Chart 3: categorical comparison

   Save them as:
   ```python
   fig.write_html("chart1.html")
   fig.write_html("chart2.html")
   fig.write_html("chart3.html")
   ```
   Use plotly.express only. No matplotlib.

4. Print a final insight report (~300 words) covering:
   - Key findings
   - What each chart shows
   - Recommendations

RULES:
- Use `import plotly.express as px` only. Never matplotlib.
- Save charts with `fig.write_html()`.
- Final answer = only the insight report text, no code.
- If you hit an error, debug and retry.
- Don't ask for input, decide yourself.
"""

    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: """ + instructions + """
Thought:{agent_scratchpad}"""

    return PromptTemplate(
        input_variables=["tools", "tool_names", "agent_scratchpad"],
        template=template,
    )


def run_agent(csv_path, df):
    """
    Run the analysis agent on a CSV file.
    Returns dict with output, charts, token stats, and compression metrics.
    """
    # --- build raw context (before compression) ---
    buf_raw = io.StringIO()
    df.info(buf=buf_raw)
    raw_info = buf_raw.getvalue()
    raw_head = df.head().to_string()
    raw_describe = df.describe(include="all").to_string()
    raw_context = raw_head + raw_info + raw_describe
    raw_tokens = count_tokens(raw_context)

    # --- build compressed context ---
    comp_head = compress_head(df)
    comp_info = compress_info(df)
    comp_describe = compress_describe(df)
    comp_context = comp_head + comp_info + comp_describe
    comp_tokens = count_tokens(comp_context)

    # compression stats
    saved_tokens = raw_tokens - comp_tokens
    compression_pct = round((saved_tokens / raw_tokens) * 100, 1) if raw_tokens > 0 else 0

    llm, llm_name = get_llm()
    prompt = build_prompt(csv_path, comp_head, comp_info, comp_describe)
    tools = [PythonREPLTool()]

    # set up token tracking
    tracker = TokenTracker()

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=15,
        verbose=True,
        return_intermediate_steps=True,
    )

    result = executor.invoke(
        {"input": "Analyse the dataset"},
        config={"callbacks": [tracker]},
    )

    chart_files = [f"chart{i}.html" for i in range(1, 4) if os.path.exists(f"chart{i}.html")]

    return {
        "output": result.get("output", "No output generated."),
        "intermediate_steps": result.get("intermediate_steps", []),
        "charts": chart_files,
        "llm_name": llm_name,
        # token tracking
        "token_stats": tracker.get_summary(),
        # compression metrics
        "compression": {
            "raw_context_tokens": raw_tokens,
            "compressed_context_tokens": comp_tokens,
            "tokens_saved": saved_tokens,
            "compression_pct": compression_pct,
        },
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agent.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    dataframe = pd.read_csv(csv_file)
    result = run_agent(csv_file, dataframe)
    print("\n" + "=" * 50)
    print("INSIGHT REPORT")
    print("=" * 50)
    print(result["output"])

    c = result["compression"]
    t = result["token_stats"]
    print(f"\n--- Token Usage ---")
    print(f"Prompt compression: {c['raw_context_tokens']} -> {c['compressed_context_tokens']} tokens ({c['compression_pct']}% saved)")
    print(f"Total LLM tokens: {t['total_tokens']} ({t['num_llm_calls']} calls)")
    print(f"Estimated cost: ${t['estimated_cost_usd']:.4f}")
    print(f"Charts: {result['charts']}")
