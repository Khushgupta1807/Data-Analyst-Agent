"""
Streamlit app — upload CSV, pick your LLM provider, run the agent.
"""

import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        color: #e8e8e8;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.25);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0b8;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .token-card {
        background: linear-gradient(135deg, #0f3443 0%, #34e89e20 100%);
        border: 1px solid rgba(52, 232, 158, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .token-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #34e89e;
    }
    .token-label {
        font-size: 0.8rem;
        color: #a0a0b8;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .llm-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00c9ff 0%, #92fe9d 100%);
        color: #1e1e2f;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    .llm-badge-inactive {
        display: inline-block;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# header
st.markdown("""
<div class="main-header">
    <h1>📊 Data Analyst Agent</h1>
    <p>Upload a CSV and let the agent analyse it — works with Groq, OpenAI, Gemini, or Claude</p>
</div>
""", unsafe_allow_html=True)

# sidebar — provider selection
from llm_config import PROVIDERS, get_llm, detect_provider

with st.sidebar:
    st.markdown("### LLM Provider")

    provider = st.selectbox(
        "Choose provider",
        list(PROVIDERS.keys()),
        index=0,
        help="Pick which LLM to use for analysis",
    )

    provider_config = PROVIDERS[provider]

    api_key = st.text_input(
        f"{provider} API Key",
        type="password",
        placeholder=f"{provider_config['key_prefix']}...",
        help=f"Enter your {provider} API key",
    )

    model = st.selectbox(
        "Model",
        provider_config["models"],
        index=0,
    )

    if api_key:
        st.markdown(
            f'<div class="llm-badge">{provider} ({model})</div>',
            unsafe_allow_html=True,
        )
        llm_available = True
    else:
        st.markdown(
            '<div class="llm-badge-inactive">No API Key</div>',
            unsafe_allow_html=True,
        )
        st.info(
            "Enter your API key above to get started."
        )
        llm_available = False

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Pick a provider & paste your key
    2. Upload a CSV file
    3. Click **Analyse**
    4. Get charts + report + token stats
    """)
    st.markdown("---")
    st.markdown("### Free API keys")
    st.markdown("""
    - [Groq](https://console.groq.com) — free tier
    - [OpenAI](https://platform.openai.com) — pay-as-you-go
    - [Google Gemini](https://aistudio.google.com) — free tier
    - [Anthropic](https://console.anthropic.com) — pay-as-you-go
    """)

# main area
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
    help="Any CSV file works"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        csv_path = uploaded_file.name
        df.to_csv(csv_path, index=False)

        st.markdown("### Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[0]:,}</div>
                <div class="metric-label">Rows</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[1]}</div>
                <div class="metric-label">Columns</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            missing = df.isnull().sum().sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{missing:,}</div>
                <div class="metric-label">Missing Values</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Analyse My Data", use_container_width=True):
            if not llm_available:
                st.error("No API key set. Pick a provider and enter your key in the sidebar.")
            else:
                with st.spinner(f"Running agent with {provider} ({model})..."):
                    try:
                        # init LLM with selected provider
                        llm, llm_name = get_llm(provider, api_key, model)

                        from agent import run_agent
                        result = run_agent(csv_path, df, llm, llm_name)

                        # token usage panel
                        st.markdown("### Token Usage & Optimization")
                        comp = result.get("compression", {})
                        tok = result.get("token_stats", {})

                        tc1, tc2, tc3, tc4 = st.columns(4)
                        with tc1:
                            st.markdown(f"""
                            <div class="token-card">
                                <div class="token-value">{comp.get('compression_pct', 0)}%</div>
                                <div class="token-label">Prompt Compressed</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with tc2:
                            st.markdown(f"""
                            <div class="token-card">
                                <div class="token-value">{tok.get('total_tokens', 0):,}</div>
                                <div class="token-label">Total Tokens</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with tc3:
                            st.markdown(f"""
                            <div class="token-card">
                                <div class="token-value">{tok.get('num_llm_calls', 0)}</div>
                                <div class="token-label">LLM Calls</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with tc4:
                            cost = tok.get('estimated_cost_usd', 0)
                            st.markdown(f"""
                            <div class="token-card">
                                <div class="token-value">${cost:.4f}</div>
                                <div class="token-label">Est. Cost</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with st.expander("Compression details"):
                            raw_tk = comp.get('raw_context_tokens', 0)
                            comp_tk = comp.get('compressed_context_tokens', 0)
                            saved = comp.get('tokens_saved', 0)

                            st.markdown(f"""
                            | Metric | Value |
                            |--------|-------|
                            | Raw context tokens | {raw_tk:,} |
                            | After compression | {comp_tk:,} |
                            | Tokens saved | {saved:,} |
                            | Compression ratio | **{comp.get('compression_pct', 0)}%** |
                            """)

                            steps_data = tok.get("per_step", [])
                            if steps_data:
                                st.markdown("**Per-step token usage:**")
                                step_df = pd.DataFrame(steps_data)
                                st.dataframe(step_df, use_container_width=True)

                        # charts
                        st.markdown("### Charts")
                        chart_titles = [
                            "Chart 1 — Distribution",
                            "Chart 2 — Relationships",
                            "Chart 3 — Comparison",
                        ]

                        charts = result.get("charts", [])
                        if charts:
                            for idx, chart_file in enumerate(charts):
                                if os.path.exists(chart_file):
                                    with open(chart_file, "r", encoding="utf-8") as f:
                                        chart_html = f.read()
                                    title = chart_titles[idx] if idx < len(chart_titles) else f"Chart {idx+1}"
                                    st.markdown(f"#### {title}")
                                    components.html(chart_html, height=450, scrolling=True)
                        else:
                            st.warning("No charts were generated.")

                        # insight report
                        st.markdown("### Insight Report")
                        st.info(result.get("output", "No output."))

                        # agent reasoning
                        with st.expander("Agent reasoning (raw steps)", expanded=False):
                            steps = result.get("intermediate_steps", [])
                            if steps:
                                for i, step in enumerate(steps):
                                    st.markdown(f"**Step {i + 1}:**")
                                    if isinstance(step, tuple) and len(step) >= 2:
                                        action, obs = step[0], step[1]
                                        st.markdown(f"**Action:** `{action.tool}`")
                                        st.code(action.tool_input, language="python")
                                        st.markdown("**Result:**")
                                        st.code(str(obs)[:2000])
                                    else:
                                        st.text(str(step)[:2000])
                                    st.markdown("---")
                            else:
                                st.text("No steps recorded.")

                        st.success("Done!")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        with st.expander("Full traceback"):
                            st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"Could not read CSV: {str(e)}")
else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #a0a0b8;">
        <h2 style="color: #667eea;">Upload a CSV to get started</h2>
        <p style="font-size: 1.1rem;">
            Pick your LLM provider in the sidebar, upload a CSV,
            and let the agent do the rest.
        </p>
    </div>
    """, unsafe_allow_html=True)
