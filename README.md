# Data Analyst Agent

A tool that takes any CSV file and runs an LLM-powered agent to clean the data, build 3 Plotly charts, and write up a summary of what it found. Supports multiple LLM providers and tracks token usage to keep costs transparent. Built with LangChain + Streamlit.

## How it works

```
                    Streamlit UI (app.py)
          ┌─────────────────────────────────┐
          │  Provider selector (sidebar)    │
          │  Groq / OpenAI / Gemini / Claude│
          │  + model picker + API key input │
          └──────────────┬──────────────────┘
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
          Selected LLM Provider
          (llm_config.py)
                   │
                   ▼
    3 Plotly charts + insight report
    + token usage dashboard
```

The agent uses the ReAct pattern — it reasons about what to do, writes Python code, executes it, reads the output, and loops until it has 3 charts and a written report.

## Supported LLM providers

| Provider | Models | Free tier? |
|----------|--------|-----------|
| Groq | llama-3.3-70b, llama-3.1-8b, mixtral-8x7b | Yes |
| OpenAI | gpt-4o-mini, gpt-4o, gpt-3.5-turbo | No (pay-as-you-go) |
| Google Gemini | gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro | Yes |
| Anthropic | claude-sonnet-4, claude-3.5-haiku | No (pay-as-you-go) |

Just pick one from the dropdown and paste your API key. The app auto-detects models for each provider.

## Setup

```bash
git clone https://github.com/Khushgupta1807/Data-Analyst-Agent.git
cd Data-Analyst-Agent

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

Open `http://localhost:8501`, pick a provider, paste your key, upload a CSV, and hit Analyse.

## Token optimization

The agent compresses the dataset context before each LLM call:

- **Row sampling** — sends 3 rows instead of 5 in the preview
- **String truncation** — caps text columns at 30 characters
- **Column pruning** — limits `describe()` to 12 columns for wide datasets
- **Info trimming** — keeps `df.info()` under 25 lines

Saves around 40-60% of context tokens. After the run, the dashboard shows total tokens used, per-step breakdown, compression ratio, and estimated cost.

## Project structure

```
├── app.py              # Streamlit frontend + token dashboard
├── agent.py            # LangChain agent + prompt compression
├── token_tracker.py    # LLM callback for tracking token usage
├── llm_config.py       # Multi-provider LLM setup
├── requirements.txt
├── .env                # API key (optional, not tracked by git)
└── README.md
```

## Tech used

- **LangChain** — ReAct agent framework
- **Groq / OpenAI / Gemini / Anthropic** — LLM inference
- **Plotly Express** — interactive charts
- **Streamlit** — web UI
- **tiktoken** — token counting
- **Pandas** — data handling
