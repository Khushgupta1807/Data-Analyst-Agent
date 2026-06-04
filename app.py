"""
Streamlit app — upload CSV, run the agent, see charts + report.
"""

import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# page config
st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# styling
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
    <p>Upload a CSV and let the agent analyse it — generates charts and a written report</p>
</div>
""", unsafe_allow_html=True)

# sidebar
with st.sidebar:
    st.markdown("### Settings")

    groq_key_input = st.text_input(
        "Groq API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Free key from console.groq.com",
        placeholder="gsk_...",
    )

    if groq_key_input:
        os.environ["GROQ_API_KEY"] = groq_key_input
        st.markdown(
            '<div class="llm-badge">Groq Cloud (llama-3.3-70b)</div>',
            unsafe_allow_html=True,
        )
        llm_available = True
    else:
        st.markdown(
            '<div class="llm-badge-inactive">No API Key</div>',
            unsafe_allow_html=True,
        )
        st.warning(
            "Enter your Groq API key above. "
            "Get a free one at [console.groq.com](https://console.groq.com)"
        )
        llm_available = False

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Paste your Groq API key
    2. Upload a CSV file
    3. Click **Analyse**
    4. Wait for the charts + report
    """)
    st.markdown("---")
    st.markdown(
        "Built with LangChain, Plotly, Streamlit"
    )

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

        # metric cards
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
                st.error(
                    "No API key set. Enter your Groq key in the sidebar first."
                )
            else:
                with st.spinner("Running agent... this takes a minute or two"):
                    try:
                        from agent import run_agent
                        result = run_agent(csv_path, df)

                        # show charts
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

                        # show agent reasoning
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
            The agent will clean the data, build 3 interactive charts,
            and write a summary report.
        </p>
    </div>
    """, unsafe_allow_html=True)
