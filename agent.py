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
from token_tracker import TokenTracker, count_tokens


# --- prompt compression ---

MAX_HEAD_ROWS = 3
MAX_COLS_IN_DESCRIBE = 12
MAX_STR_COL_WIDTH = 30
MAX_INFO_LINES = 25

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
    if len(lines) > MAX_INFO_LINES:
        kept = lines[:5] + lines[5:MAX_INFO_LINES-2] + ["...", lines[-2]]
        return "\n".join(kept)
    return "\n".join(lines)

def compress_describe(df):
    """If too many columns, only describe the most interesting ones."""
    if df.shape[1] > MAX_COLS_IN_DESCRIBE:
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

3. Create exactly 7 Plotly charts showing the most interesting patterns.
   Use plotly.express AND plotly.graph_objects for styling.
   
   DEFINE THIS COLOR PALETTE at the top of your code:
   ```python
   import plotly.express as px
   import plotly.graph_objects as go
   COLORS = ["#667eea", "#764ba2", "#34e89e", "#00c9ff", "#f093fb", "#ffd166", "#ff6b6b"]
   ```
   
   Choose chart types that best fit the data. Use DIFFERENT types:
   - Chart 1: Donut/pie chart for composition (e.g. category distribution)
   - Chart 2: Horizontal bar chart for rankings/comparisons
   - Chart 3: Scatter plot for correlations between two numeric columns
   - Chart 4: Heatmap showing correlation matrix of numeric columns
   - Chart 5: Line or area chart for trends (or grouped bar chart)
   - Chart 6: Box plot or violin plot for distribution analysis
   - Chart 7: Histogram for deeper analysis

   ⚠️ CRITICAL STYLING RULES — YOU MUST FOLLOW ALL OF THESE:

   A) For EVERY plotly.express chart, pass `color_discrete_sequence=COLORS`:
      ```python
      fig = px.bar(..., color_discrete_sequence=COLORS)
      fig = px.scatter(..., color_discrete_sequence=COLORS)
      fig = px.box(..., color_discrete_sequence=COLORS)
      fig = px.pie(..., color_discrete_sequence=COLORS)
      fig = px.histogram(..., color_discrete_sequence=COLORS)
      ```

   B) For EVERY graph_objects trace, set marker color explicitly:
      ```python
      go.Bar(marker=dict(color="#667eea"))
      go.Scatter(line=dict(color="#667eea", width=2))
      go.Box(marker=dict(color="#667eea"), line=dict(color="#764ba2"))
      ```

   C) NEVER use black, #000000, #000001 or any dark color for markers/bars/lines.
      ALL data elements MUST use bright, vibrant colors from COLORS.

   D) For box plots, always use the `color` parameter:
      ```python
      fig = px.box(df, x="category_col", y="numeric_col", color="category_col",
                   color_discrete_sequence=COLORS)
      ```

   E) For bar charts, always use the `color` parameter:
      ```python
      fig = px.bar(df, x="col", y="col", color="col",
                   color_discrete_sequence=COLORS)
      ```

   F) Apply this layout to EVERY chart after creating it:
      ```python
      fig.update_layout(
          template="plotly_dark",
          paper_bgcolor="rgba(30,30,47,1)",
          plot_bgcolor="rgba(30,30,47,1)",
          font=dict(family="Inter, sans-serif", color="#e0e0e0"),
          title=dict(font=dict(size=18, color="white")),
          margin=dict(l=40, r=40, t=60, b=40),
          legend=dict(bgcolor="rgba(0,0,0,0)"),
      )
      ```

   G) Give EVERY chart a descriptive title using `title="Chart Title"` in the px call.

   Save them as:
   ```python
   fig.write_html("chart1.html", include_plotlyjs="cdn")
   fig.write_html("chart2.html", include_plotlyjs="cdn")
   fig.write_html("chart3.html", include_plotlyjs="cdn")
   fig.write_html("chart4.html", include_plotlyjs="cdn")
   fig.write_html("chart5.html", include_plotlyjs="cdn")
   fig.write_html("chart6.html", include_plotlyjs="cdn")
   fig.write_html("chart7.html", include_plotlyjs="cdn")
   ```

4. Print a final insight report (~300 words) covering:
   - Key findings
   - What each chart shows
   - Recommendations

RULES:
- Use plotly.express and plotly.graph_objects. Never matplotlib.
- Apply the dark styling above to ALL charts.
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


def run_agent(csv_path, df, llm, llm_name, provider="Groq"):
    """
    Run the analysis agent on a CSV file.
    Takes llm and llm_name directly (provider chosen in the UI).
    Returns dict with output, charts, token stats, and compression metrics.
    """
    # raw context (before compression) for comparison
    buf_raw = io.StringIO()
    df.info(buf=buf_raw)
    raw_context = df.head().to_string() + buf_raw.getvalue() + df.describe(include="all").to_string()
    raw_tokens = count_tokens(raw_context)

    # compressed context
    comp_head = compress_head(df)
    comp_info = compress_info(df)
    comp_describe = compress_describe(df)
    comp_context = comp_head + comp_info + comp_describe
    comp_tokens = count_tokens(comp_context)

    saved_tokens = raw_tokens - comp_tokens
    compression_pct = round((saved_tokens / raw_tokens) * 100, 1) if raw_tokens > 0 else 0

    prompt = build_prompt(csv_path, comp_head, comp_info, comp_describe)
    tools = [PythonREPLTool()]
    tracker = TokenTracker(provider=provider)

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=20,
        verbose=True,
        return_intermediate_steps=True,
    )

    result = executor.invoke(
        {"input": "Analyse the dataset"},
        config={"callbacks": [tracker]},
    )

    chart_files = [f"chart{i}.html" for i in range(1, 8) if os.path.exists(f"chart{i}.html")]

    return {
        "output": result.get("output", "No output generated."),
        "intermediate_steps": result.get("intermediate_steps", []),
        "charts": chart_files,
        "llm_name": llm_name,
        "token_stats": tracker.get_summary(),
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

    from llm_config import get_llm
    csv_file = sys.argv[1]
    dataframe = pd.read_csv(csv_file)

    # default to groq for CLI usage
    import os
    api_key = os.environ.get("GROQ_API_KEY", "")
    llm, name = get_llm("Groq", api_key)
    result = run_agent(csv_file, dataframe, llm, name)

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
