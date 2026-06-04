"""
Main agent logic — builds a ReAct agent that can write + run Python code
to analyse a CSV, create plotly charts, and produce an insight report.
"""

import io
import os
import pandas as pd
from langchain_experimental.tools import PythonREPLTool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from llm_config import get_llm


def build_prompt(csv_path, df_head, df_info, df_describe):
    """Build the full ReAct prompt with dataset context baked in."""

    instructions = f"""You are an expert data analyst. You have access to a Python REPL tool.
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
    Returns dict with 'output', 'intermediate_steps', 'charts', 'llm_name'.
    """
    # grab dataset info as strings for the prompt
    buf = io.StringIO()
    df.info(buf=buf)
    df_info = buf.getvalue()
    df_head = df.head().to_string()
    df_describe = df.describe(include="all").to_string()

    llm, llm_name = get_llm()
    prompt = build_prompt(csv_path, df_head, df_info, df_describe)
    tools = [PythonREPLTool()]

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=15,
        verbose=True,
        return_intermediate_steps=True,
    )

    result = executor.invoke({"input": "Analyse the dataset"})

    # check which chart files got created
    chart_files = [f"chart{i}.html" for i in range(1, 4) if os.path.exists(f"chart{i}.html")]

    return {
        "output": result.get("output", "No output generated."),
        "intermediate_steps": result.get("intermediate_steps", []),
        "charts": chart_files,
        "llm_name": llm_name,
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
    print(f"\nCharts: {result['charts']}")
