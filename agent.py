"""
Autonomous Data Analyst Agent
Uses LangChain ReAct agent with PythonREPLTool to analyse CSV data,
generate Plotly charts, and produce insight reports.
"""

import io
import os
import pandas as pd
from langchain_experimental.tools import PythonREPLTool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from llm_config import get_llm


# ──────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────

def build_prompt(csv_path: str, df_head: str, df_info: str, df_describe: str) -> PromptTemplate:
    """
    Constructs the system prompt that instructs the agent on
    what analysis to perform and what output to produce.
    """

    system_instructions = f"""You are an expert data analyst. You have access to a Python REPL tool.
Your job is to analyse a CSV dataset and produce visualisations and insights.

DATASET LOCATION: {csv_path}
The file is already on disk — load it with pandas.

Here is a preview of the dataset:

--- HEAD (first 5 rows) ---
{df_head}

--- INFO ---
{df_info}

--- DESCRIBE ---
{df_describe}

YOUR TASK (follow these steps in order):

1. **Load the CSV** into a pandas DataFrame using `pd.read_csv('{csv_path}')`.

2. **Clean the data**: handle missing values appropriately (drop or fill depending on column type).

3. **Create exactly 3 Plotly Express charts** that reveal the most interesting patterns:
   - Chart 1: A distribution or histogram chart
   - Chart 2: A relationship or correlation chart (scatter, bar, etc.)
   - Chart 3: A categorical comparison or composition chart

   Save each chart as an HTML file:
   ```python
   fig.write_html("chart1.html")
   fig.write_html("chart2.html")
   fig.write_html("chart3.html")
   ```
   IMPORTANT: Import plotly.express as px. Do NOT use matplotlib.

4. **Print a final insight report** (approximately 300 words) summarizing:
   - Key findings from the data
   - What each chart reveals
   - Actionable recommendations

RULES:
- Always use `import plotly.express as px` — NEVER use matplotlib.
- Always save charts with `fig.write_html()`.
- Your FINAL answer must be ONLY the insight report text (no code).
- If you encounter an error, debug it and try again.
- Do not ask for human input. Make all decisions yourself.
"""

    react_template = """Answer the following questions as best you can. You have access to the following tools:

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

Question: """ + system_instructions + """
Thought:{agent_scratchpad}"""

    return PromptTemplate(
        input_variables=["tools", "tool_names", "agent_scratchpad"],
        template=react_template,
    )


# ──────────────────────────────────────────────
# Agent Runner
# ──────────────────────────────────────────────

def run_agent(csv_path: str, df: pd.DataFrame) -> dict:
    """
    Runs the data-analysis agent on the given CSV.

    Returns:
        dict with keys:
            - output: the insight report text
            - intermediate_steps: raw agent reasoning steps
            - charts: list of chart file paths that were generated
    """

    # Capture df.info() as string
    buf = io.StringIO()
    df.info(buf=buf)
    df_info = buf.getvalue()

    df_head = df.head().to_string()
    df_describe = df.describe(include="all").to_string()

    # Get LLM
    llm, llm_name = get_llm()

    # Build prompt
    prompt = build_prompt(csv_path, df_head, df_info, df_describe)

    # Tools
    tools = [PythonREPLTool()]

    # Create ReAct agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # Agent executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=15,
        verbose=True,
        return_intermediate_steps=True,
    )

    # Run
    result = executor.invoke({"input": "Analyse the dataset"})

    # Collect generated chart files
    chart_files = []
    for i in range(1, 4):
        chart_path = f"chart{i}.html"
        if os.path.exists(chart_path):
            chart_files.append(chart_path)

    return {
        "output": result.get("output", "No output generated."),
        "intermediate_steps": result.get("intermediate_steps", []),
        "charts": chart_files,
        "llm_name": llm_name,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agent.py <path_to_csv>")
        sys.exit(1)
    csv_file = sys.argv[1]
    dataframe = pd.read_csv(csv_file)
    result = run_agent(csv_file, dataframe)
    print("\n" + "=" * 60)
    print("INSIGHT REPORT")
    print("=" * 60)
    print(result["output"])
    print(f"\nCharts generated: {result['charts']}")
