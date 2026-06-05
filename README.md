# Data Analyst Agent

An autonomous AI-powered data analysis tool that takes any CSV dataset and produces 7 interactive Plotly visualizations, a written insight report, and full token usage analytics — all in one click. Supports 4 LLM providers (Groq, OpenAI, Google Gemini, Anthropic) with real-time cost tracking and prompt compression that saves 40–60% of context tokens. Built with LangChain ReAct agents + Streamlit.

## Features

- **Autonomous analysis** — upload a CSV and the agent decides what to clean, chart, and report
- **7 interactive charts** — donut, bar, scatter, heatmap, line/area, box/violin, treemap/sunburst
- **4 LLM providers** — switch between Groq (free), OpenAI, Google Gemini (free), or Anthropic
- **Token optimization** — compresses dataset context before sending to the LLM, saving 40–60% tokens
- **Cost tracking** — per-step token breakdown with provider-specific cost estimation
- **Dark-themed dashboard** — professional BI-style layout with KPI cards and 2-column chart grid
- **Agent reasoning** — inspect each step the agent took (action, code, result)

## Architecture

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
              (compress_head, compress_info,
               compress_describe)
                   │
                   ▼
          Selected LLM Provider
          (llm_config.py)
                   │
                   ▼
          Token Tracker Callback
          (token_tracker.py)
                   │
                   ▼
    7 Plotly charts + insight report
    + token usage dashboard + cost estimate
```

### How the ReAct loop works

1. The agent receives the compressed dataset preview (head, info, describe)
2. It **thinks** about what analysis to perform
3. It **writes Python code** and executes it in a sandboxed REPL
4. It **reads the output** (success, error, or data)
5. It **loops** — repeating steps 2–4 until all 7 charts are saved and a report is written
6. The final answer is the insight report text (no code)

Typical run: 5–12 LLM calls, 3,000–8,000 total tokens, under $0.01 cost on Groq.

## Supported LLM providers

| Provider | Models | Free tier? | Cost (per 1M tokens) |
|----------|--------|------------|---------------------|
| Groq | llama-3.3-70b, llama-3.1-8b, mixtral-8x7b | Yes | $0.59 input / $0.79 output |
| OpenAI | gpt-4o-mini, gpt-4o, gpt-3.5-turbo | No (pay-as-you-go) | $0.15 input / $0.60 output |
| Google Gemini | gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro | Yes | $0.075 input / $0.30 output |
| Anthropic | claude-sonnet-4, claude-3.5-haiku | No (pay-as-you-go) | $3.00 input / $15.00 output |

Pick one from the dropdown and paste your API key. The app auto-detects available models for each provider.

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

Open `http://localhost:8501`, pick a provider, paste your key, upload a CSV, and hit **Analyse**.

## Token optimization

The agent compresses the dataset context before each LLM call using 4 techniques:

| Technique | What it does | Token savings |
|-----------|-------------|---------------|
| Row sampling | Sends 3 rows instead of 5 in preview | ~20% |
| String truncation | Caps text columns at 30 characters | ~15% |
| Column pruning | Limits `describe()` to 12 columns for wide datasets | ~25% |
| Info trimming | Keeps `df.info()` under 25 lines | ~10% |

**Combined savings: 40–60%** of context tokens on a typical dataset.

After each run, the dashboard shows:
- Raw vs compressed token counts (before/after)
- Compression percentage saved
- Input and output tokens per LLM call
- Estimated cost in USD (provider-specific pricing)

## Dashboard sections

1. **KPI strip** — rows, columns, numeric columns, categorical columns, missing data %
2. **Data preview** — first 5 rows of the uploaded CSV
3. **Token optimization panel** — before/after compression, savings %, input/output tokens, cost
4. **Per-step breakdown** — expandable table showing tokens used in each LLM call
5. **Visualizations** — 7 interactive Plotly charts in a 2-column grid (dark themed)
6. **Insight report** — AI-generated analysis findings and recommendations
7. **Dataset summary** — full `describe()` statistics for all columns
8. **Agent reasoning** — expandable raw steps showing the agent's thought process

## Project structure

```
├── app.py              # Streamlit frontend + BI dashboard layout
├── agent.py            # LangChain ReAct agent + prompt compression
├── token_tracker.py    # LLM callback for tracking token usage + cost
├── llm_config.py       # Multi-provider LLM setup (Groq/OpenAI/Gemini/Anthropic)
├── requirements.txt    # Python dependencies
├── .env                # API key (optional, not tracked by git)
├── .gitignore          # Excludes venv, .env, generated charts
└── README.md
```

## Tech stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Agent framework | LangChain (ReAct) | Autonomous reasoning + tool use |
| LLM inference | Groq / OpenAI / Gemini / Anthropic | Natural language understanding |
| Visualizations | Plotly Express + Graph Objects | 7 interactive dark-themed charts |
| Web UI | Streamlit | Dashboard with KPI cards, charts, tables |
| Token counting | tiktoken | Accurate token estimation |
| Data handling | Pandas | CSV parsing, cleaning, statistics |
| Cost tracking | Custom TokenTracker | Per-step token + cost breakdown |
