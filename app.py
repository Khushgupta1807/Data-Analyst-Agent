"""
Streamlit app — upload CSV, pick your LLM provider, run the agent.
BI dashboard layout inspired by professional data analytics tools.
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* hide streamlit defaults — keep header so sidebar toggle works */
    .stDeployButton, #MainMenu, footer {
        display: none !important;
    }

    /* make header transparent but keep sidebar toggle visible */
    header[data-testid="stHeader"] {
        background: transparent !important;
        border: none !important;
    }

    /* style the sidebar collapse/expand toggle so it's always visible */
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="collapsedControl"] {
        color: #667eea !important;
        background: rgba(30, 30, 47, 0.9) !important;
        border: 1px solid rgba(102, 126, 234, 0.4) !important;
        border-radius: 8px !important;
        width: 2.2rem !important;
        height: 2.2rem !important;
        z-index: 999 !important;
    }
    button[data-testid="stSidebarCollapseButton"]:hover,
    button[data-testid="collapsedControl"]:hover {
        background: rgba(102, 126, 234, 0.3) !important;
        transform: scale(1.05);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* header banner */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        color: white;
    }
    .main-header p {
        font-size: 0.95rem;
        opacity: 0.85;
        margin: 0.3rem 0 0 0;
        color: #e8e8e8;
    }

    /* KPI cards — top strip */
    .kpi-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #667eea;
        line-height: 1.2;
        text-decoration: none !important;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #8888a8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.2rem;
    }

    /* token optimization cards */
    .token-card {
        background: linear-gradient(135deg, #0f3443 0%, #1a4a3a 100%);
        border: 1px solid rgba(52, 232, 158, 0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .token-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #34e89e;
        line-height: 1.2;
        text-decoration: none !important;
    }
    .kpi-card *, .token-card * {
        text-decoration: none !important;
    }
    .token-label {
        font-size: 0.7rem;
        color: #8888a8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 0.2rem;
    }

    /* chart containers */
    .chart-box {
        background: #1e1e2f;
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 12px;
        padding: 0.5rem;
        margin-bottom: 0.8rem;
    }

    /* section titles */
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #c0c0d8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 1.2rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }

    /* sidebar badges */
    .llm-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00c9ff 0%, #92fe9d 100%);
        color: #1e1e2f;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-top: 0.4rem;
    }
    .llm-badge-inactive {
        display: inline-block;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-top: 0.4rem;
    }

    /* analyse button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        font-weight: 700;
        transition: all 0.3s ease;
        width: 100%;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* data table styling */
    .dataframe-container {
        background: #1e1e2f;
        border-radius: 12px;
        padding: 0.8rem;
        border: 1px solid rgba(102, 126, 234, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# header
st.markdown("""
<div class="main-header">
    <div>
        <h1>📊 Data Analyst Agent</h1>
        <p>Upload a CSV — get charts, insights, and token analytics</p>
    </div>
</div>
""", unsafe_allow_html=True)

# sidebar
from llm_config import PROVIDERS, get_llm

with st.sidebar:
    st.markdown("### LLM Provider")

    provider = st.selectbox(
        "Choose provider",
        list(PROVIDERS.keys()),
        index=0,
    )
    provider_config = PROVIDERS[provider]

    api_key = st.text_input(
        f"{provider} API Key",
        type="password",
        placeholder=f"{provider_config['key_prefix']}...",
    )

    model = st.selectbox("Model", provider_config["models"], index=0)

    if api_key:
        st.markdown(f'<div class="llm-badge">{provider} ({model})</div>', unsafe_allow_html=True)
        llm_available = True
    else:
        st.markdown('<div class="llm-badge-inactive">No API Key</div>', unsafe_allow_html=True)
        st.info("Enter your API key above.")
        llm_available = False

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Pick a provider & paste key
    2. Upload a CSV
    3. Click **Analyse**
    """)
    st.markdown("---")
    st.markdown("### Free keys")
    st.markdown("""
    - [Groq](https://console.groq.com) — free
    - [Gemini](https://aistudio.google.com) — free
    - [OpenAI](https://platform.openai.com)
    - [Anthropic](https://console.anthropic.com)
    """)

# main area
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        csv_path = uploaded_file.name
        df.to_csv(csv_path, index=False)

        # --- KPI strip (like the reference dashboard top row) ---
        num_cols = len(df.select_dtypes(include=["number"]).columns)
        cat_cols = len(df.select_dtypes(include=["object", "category"]).columns)
        missing = df.isnull().sum().sum()
        missing_pct = round((missing / (df.shape[0] * df.shape[1])) * 100, 1)

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value">{df.shape[0]:,}</div>
                <div class="kpi-label">Rows</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value">{df.shape[1]}</div>
                <div class="kpi-label">Columns</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value">{num_cols}</div>
                <div class="kpi-label">Numeric</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value">{cat_cols}</div>
                <div class="kpi-label">Categorical</div>
            </div>""", unsafe_allow_html=True)
        with k5:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value">{missing_pct}%</div>
                <div class="kpi-label">Missing</div>
            </div>""", unsafe_allow_html=True)

        # data preview in a styled container
        st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(5), use_container_width=True, height=200)

        # analyse button
        if st.button("Analyse My Data", use_container_width=True):
            if not llm_available:
                st.error("No API key set. Pick a provider in the sidebar.")
            else:
                with st.spinner(f"Running agent with {provider} ({model})..."):
                    try:
                        llm, llm_name = get_llm(provider, api_key, model)
                        from agent import run_agent
                        result = run_agent(csv_path, df, llm, llm_name, provider=provider)

                        # --- TOKEN OPTIMIZATION (before/after strip) ---
                        st.markdown('<div class="section-title">Token Optimization</div>', unsafe_allow_html=True)
                        comp = result.get("compression", {})
                        tok = result.get("token_stats", {})

                        raw_tk = comp.get('raw_context_tokens', 0)
                        comp_tk = comp.get('compressed_context_tokens', 0)
                        pct = comp.get('compression_pct', 0)

                        o1, o2, o3, o4, o5, o6 = st.columns(6)
                        with o1:
                            st.markdown(f"""<div class="token-card">
                                <div class="token-value" style="color:#ff6b6b;">{raw_tk:,}</div>
                                <div class="token-label">Before</div>
                            </div>""", unsafe_allow_html=True)
                        with o2:
                            st.markdown(f"""<div class="token-card">
                                <div class="token-value">{comp_tk:,}</div>
                                <div class="token-label">After</div>
                            </div>""", unsafe_allow_html=True)
                        with o3:
                            st.markdown(f"""<div class="token-card">
                                <div class="token-value">{pct}%</div>
                                <div class="token-label">Saved</div>
                            </div>""", unsafe_allow_html=True)
                        with o4:
                            st.markdown(f"""<div class="token-card">
                                <div class="token-value">{tok.get('total_input_tokens',0):,}</div>
                                <div class="token-label">Input Tok</div>
                            </div>""", unsafe_allow_html=True)
                        with o5:
                            st.markdown(f"""<div class="token-card">
                                <div class="token-value">{tok.get('total_output_tokens',0):,}</div>
                                <div class="token-label">Output Tok</div>
                            </div>""", unsafe_allow_html=True)
                        with o6:
                            cost = tok.get('estimated_cost_usd', 0)
                            st.markdown(f"""<div class="token-card">
                                <div class="token-value">${cost:.4f}</div>
                                <div class="token-label">Cost ({provider})</div>
                            </div>""", unsafe_allow_html=True)

                        # per-step breakdown
                        with st.expander(f"Per-step breakdown ({tok.get('num_llm_calls',0)} LLM calls)"):
                            steps_data = tok.get("per_step", [])
                            if steps_data:
                                st.dataframe(pd.DataFrame(steps_data), use_container_width=True)
                            else:
                                st.text("No per-step data.")

                        # --- CHARTS in 2-column grid ---
                        st.markdown('<div class="section-title">Visualizations</div>', unsafe_allow_html=True)
                        charts = result.get("charts", [])

                        def fix_chart_html(html_content, px_height=430):
                            """Inject CSS to force Plotly chart to fill the iframe height."""
                            size_fix = f"""<style>
                                html, body {{ margin:0; padding:0; height:{px_height}px; overflow:hidden; }}
                                .plotly-graph-div {{ height:{px_height}px !important; width:100% !important; }}
                            </style>"""
                            return html_content.replace("<head>", f"<head>{size_fix}", 1)

                        if charts:
                            # display charts in pairs (2-column grid)
                            for i in range(0, len(charts), 2):
                                if i + 1 < len(charts):
                                    # two charts side by side
                                    left, right = st.columns(2)
                                    with left:
                                        if os.path.exists(charts[i]):
                                            with open(charts[i], "r", encoding="utf-8") as f:
                                                st.markdown('<div class="chart-box">', unsafe_allow_html=True)
                                                components.html(fix_chart_html(f.read(), 430), height=450, scrolling=False)
                                                st.markdown('</div>', unsafe_allow_html=True)
                                    with right:
                                        if os.path.exists(charts[i + 1]):
                                            with open(charts[i + 1], "r", encoding="utf-8") as f:
                                                st.markdown('<div class="chart-box">', unsafe_allow_html=True)
                                                components.html(fix_chart_html(f.read(), 430), height=450, scrolling=False)
                                                st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    # last chart full width if odd number
                                    if os.path.exists(charts[i]):
                                        with open(charts[i], "r", encoding="utf-8") as f:
                                            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
                                            components.html(fix_chart_html(f.read(), 480), height=500, scrolling=False)
                                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("No charts were generated.")

                        # --- INSIGHT REPORT ---
                        st.markdown('<div class="section-title">Insight Report</div>', unsafe_allow_html=True)
                        st.info(result.get("output", "No output."))

                        # --- DATA TABLE (like the reference dashboard bottom) ---
                        st.markdown('<div class="section-title">Dataset Summary</div>', unsafe_allow_html=True)
                        summary_df = df.describe(include="all").T
                        summary_df.index.name = "Column"
                        st.dataframe(summary_df, use_container_width=True, height=300)

                        # agent reasoning
                        with st.expander("Agent reasoning (raw steps)"):
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

                        st.success("Analysis complete!")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        with st.expander("Full traceback"):
                            st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"Could not read CSV: {str(e)}")
else:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 1rem; color: #8888a8;">
        <h2 style="color: #667eea; font-weight: 800;">Upload a CSV to get started</h2>
        <p style="font-size: 1rem;">
            Pick your LLM provider, upload a dataset, and the agent handles the rest.
        </p>
    </div>
    """, unsafe_allow_html=True)
