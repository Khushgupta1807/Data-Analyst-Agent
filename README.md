# 📊 Autonomous Data Analyst Agent

An AI-powered data analysis tool that autonomously cleans, visualises, and generates insights from any CSV dataset.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)              │
│  ┌───────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ CSV Upload│  │  Metric  │  │  Chart Display    │  │
│  │ & Preview │  │  Cards   │  │  (Plotly HTML)    │  │
│  └─────┬─────┘  └──────────┘  └──────────────────┘  │
│        │                                             │
│        ▼                                             │
│  ┌─────────────────────────────────────────────┐     │
│  │         Agent Executor (agent.py)           │     │
│  │  ┌──────────────┐  ┌────────────────────┐   │     │
│  │  │  ReAct Agent │  │  PythonREPL Tool   │   │     │
│  │  │  (LangChain) │──│  (Code Execution)  │   │     │
│  │  └──────┬───────┘  └────────────────────┘   │     │
│  │         │                                   │     │
│  │         ▼                                   │     │
│  │  ┌──────────────────────────────────┐       │     │
│  │  │  LLM Backend (llm_config.py)     │       │     │
│  │  │  ┌─────────┐  ┌──────────────┐   │       │     │
│  │  │  │  Groq   │  │   Ollama     │   │       │     │
│  │  │  │ (Cloud) │  │  (Local)     │   │       │     │
│  │  │  └─────────┘  └──────────────┘   │       │     │
│  │  └──────────────────────────────────┘       │     │
│  └─────────────────────────────────────────────┘     │
│                                                      │
│  Output: 3 Plotly Charts + 300-word Insight Report   │
└──────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
data-analyst-agent/
├── app.py              # Streamlit web interface
├── agent.py            # LangChain ReAct agent logic
├── llm_config.py       # LLM auto-detection (Groq / Ollama)
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (create manually)
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
cd data-analyst-agent
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure LLM

**Option A — Groq Cloud (Recommended, Free & Fast):**
1. Get a free API key at [console.groq.com](https://console.groq.com)
2. Create a `.env` file:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```

**Option B — Ollama Local:**
1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull the model: `ollama pull llama3.2`
3. No API key needed — runs locally

### 3. Launch

```bash
streamlit run app.py --server.headless true
```

Open `http://localhost:8501` in your browser.

## 🎯 Features

- **Auto-LLM Detection**: Automatically uses Groq Cloud or falls back to Ollama
- **Smart Data Cleaning**: Handles missing values automatically
- **3 Interactive Charts**: Distribution, Correlation, and Comparison visualisations
- **AI Insight Report**: ~300-word summary with actionable recommendations
- **Agent Transparency**: View the full reasoning process in an expandable section
- **Error Recovery**: Built-in error handling with detailed tracebacks

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM Framework | LangChain |
| Agent Type | ReAct (Reasoning + Acting) |
| Code Execution | PythonREPLTool |
| Visualisation | Plotly Express |
| Web UI | Streamlit |
| Cloud LLM | Groq (Llama 3.3 70B) |
| Local LLM | Ollama (Llama 3.2) |

## 📝 License

MIT License — free for personal and commercial use.
