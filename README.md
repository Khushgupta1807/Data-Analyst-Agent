# Data Analyst Agent

A tool that takes any CSV file and runs an LLM-powered agent to clean the data, build 3 Plotly charts, and write up a summary of what it found. Includes token tracking and prompt compression to keep API costs low. Built with LangChain + Streamlit.

## How it works

```
                    Streamlit UI (app.py)
                         │
                    Upload CSV file
                         │
                         ▼
              Agent Executor (agent.py)
              ┌──────────┬──────────┐
              │ ReAct    │ Python   │
              │ Agent    │ REPL     │
              └────┬─────┴──────────┘
                   │
              Prompt Compression
              (token_tracker.py)
                   │
                   ▼
            Groq Cloud API
         (Llama 3.3 70B model)
                   │
                   ▼
    3 Plotly charts + insight report
    + token usage dashboard
```

The agent uses the ReAct pattern — it reasons about what to do, writes Python code, executes it, reads the output, and loops until it has 3 charts and a written report.

Before hitting the API, the prompt goes through a compression step that trims the dataset context (fewer rows, capped string lengths, column pruning for wide datasets) to cut token usage without losing important info.

## Setup

```bash
git clone https://github.com/Khushgupta1807/Data-Analyst-Agent.git
cd Data-Analyst-Agent

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

You'll need a Groq API key (free):
1. Sign up at [console.groq.com](https://console.groq.com)
2. Create an API key
3. You can paste it directly into the app sidebar, or put it in a `.env` file:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```

## Running

```bash
streamlit run app.py
```

Then open `http://localhost:8501`, paste your API key in the sidebar, upload a CSV, and hit "Analyse".

## Token optimization

The agent compresses the dataset context before sending it to the LLM:

- **Row sampling** — sends 3 rows instead of 5 in the preview
- **String truncation** — caps text columns at 30 characters
- **Column pruning** — limits `describe()` to 12 columns for wide datasets
- **Info trimming** — keeps `df.info()` under 25 lines

This typically saves **40-60% of context tokens** depending on the dataset. The dashboard shows exactly how many tokens were used, per-step breakdowns, and estimated cost.

## Project structure

```
├── app.py              # Streamlit frontend + token dashboard
├── agent.py            # LangChain agent + prompt compression
├── token_tracker.py    # LLM callback for tracking token usage
├── llm_config.py       # Groq API setup
├── requirements.txt
├── .env                # API key goes here (not tracked by git)
└── README.md
```

## Tech used

- **LangChain** — ReAct agent framework
- **Groq** — LLM inference (Llama 3.3 70B)
- **Plotly Express** — interactive charts
- **Streamlit** — web UI
- **tiktoken** — token counting for compression metrics
- **Pandas** — data handling
