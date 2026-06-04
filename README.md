# Data Analyst Agent

A tool that takes any CSV file and runs an LLM-powered agent to clean the data, build 3 Plotly charts, and write up a summary of what it found. Built with LangChain + Streamlit.

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
                   ▼
            Groq Cloud API
         (Llama 3.3 70B model)
                   │
                   ▼
    3 Plotly charts + insight report
```

The agent uses the ReAct pattern — it thinks about what to do, writes Python code, runs it, checks the output, and repeats until it has 3 charts and a written report.

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

## Project structure

```
├── app.py              # Streamlit frontend
├── agent.py            # LangChain agent + prompt
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
- **Pandas** — data handling
