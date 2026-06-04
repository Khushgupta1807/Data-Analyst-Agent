# 📊 Autonomous Data Analyst Agent

An AI-powered data analysis tool that autonomously cleans, visualises, and generates insights from any CSV dataset. Powered by **Groq Cloud** (Llama 3.3 70B) and **LangChain ReAct agents**.

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
│  │  │     Groq Cloud (llm_config.py)   │       │     │
│  │  │     Llama 3.3 70B Versatile      │       │     │
│  │  └──────────────────────────────────┘       │     │
│  └─────────────────────────────────────────────┘     │
│                                                      │
│  Output: 3 Plotly Charts + 300-word Insight Report   │
└──────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
data-analyst-agent/
├── app.py              # Streamlit web interface with API key input
├── agent.py            # LangChain ReAct agent logic
├── llm_config.py       # LLM configuration (Groq Cloud)
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (optional)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Khushgupta1807/Data-Analyst-Agent.git
cd Data-Analyst-Agent
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Get a Groq API Key (Free)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to **API Keys** and create a new key
4. Copy the key — you'll paste it into the app

> **Note:** Groq provides free API access with generous rate limits. No credit card required.

### 3. Launch

```bash
streamlit run app.py --server.headless true
```

Open `http://localhost:8501` in your browser.

### 4. Use the App

1. Paste your **Groq API key** in the sidebar
2. Upload any **CSV file**
3. Preview the data and metrics
4. Click **"🚀 Analyse My Data"**
5. View the 3 interactive charts and insight report

## 🎯 Features

- **Groq Cloud LLM**: Uses Llama 3.3 70B via Groq for fast, high-quality analysis
- **In-App API Key Input**: Enter your Groq key directly in the sidebar — no config files needed
- **Smart Data Cleaning**: Handles missing values automatically
- **3 Interactive Plotly Charts**: Distribution, Correlation, and Categorical Comparison
- **AI Insight Report**: ~300-word summary with key findings and actionable recommendations
- **Agent Transparency**: Expandable section showing the full reasoning process step by step
- **Error Recovery**: Built-in error handling with detailed tracebacks

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM Framework | LangChain |
| Agent Type | ReAct (Reasoning + Acting) |
| Code Execution | PythonREPLTool |
| Visualisation | Plotly Express |
| Web UI | Streamlit |
| LLM Provider | Groq Cloud (Llama 3.3 70B Versatile) |

## 🔧 Configuration

### Option A — In-App (Recommended)
Paste your Groq API key directly into the sidebar input field when you launch the app.

### Option B — Environment Variable
Create a `.env` file in the project root:
```
GROQ_API_KEY=gsk_your_key_here
```

The app will auto-detect the key on startup.

## 📝 License

MIT License — free for personal and commercial use.
