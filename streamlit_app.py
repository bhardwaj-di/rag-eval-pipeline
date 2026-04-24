import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(__file__))

for _key in ["QDRANT_URL", "QDRANT_API_KEY", "GROQ_API_KEY", "NOMIC_API_KEY"]:
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

from rag.pipeline import answer_question

st.set_page_config(
    page_title="SEC 10-K Filings Assistant",
    page_icon="📈",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #1a1a4e, #24243e);
        background-attachment: fixed;
    }

    /* Main text readable on dark bg */
    .stApp, .stApp p, .stApp li, .stApp label,
    .stMarkdown, .stMarkdown p { color: #e8eaf0 !important; }

    /* Chat messages */
    .stChatMessage { border-radius: 12px; margin-bottom: 0.5rem; }
    [data-testid="stChatMessageContent"] p { color: #e8eaf0 !important; }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.07) !important;
        color: #e8eaf0 !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95) !important;
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #94a3b8 !important; }

    /* Info box */
    .stAlert { background: rgba(30,80,160,0.25) !important; border: 1px solid rgba(100,150,255,0.2) !important; }
    .stAlert p { color: #93c5fd !important; }

    .source-badge {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        color: #7eb8f7;
    }
    .score-text { color: #aaa; font-size: 12px; }

    .welcome-banner {
        text-align: center;
        padding: 2rem 1rem 1.2rem 1rem;
        background: linear-gradient(90deg, rgba(30,80,160,0.35), rgba(90,30,120,0.35));
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.5rem;
    }
    .welcome-banner h1 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.3rem 0;
        letter-spacing: 0.5px;
    }
    .welcome-banner p {
        color: #a0b8d8;
        font-size: 0.95rem;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="welcome-banner">
    <h1>📊 Welcome to Your Finance 10-K Filings Assistant</h1>
    <p>Explore annual SEC filings for AAPL, GOOGL, META, MSI & NVDA &nbsp;·&nbsp; Fiscal Years 2024–2025</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# 📈 SEC 10-K")
    st.markdown("### Filings Assistant")
    st.caption("Powered by RAG · Groq · Qdrant")

    st.markdown("---")
    st.markdown("**🏢 Filter by Company**")
    ticker_options = ["All", "AAPL", "GOOGL", "META", "MSI", "NVDA"]
    selected_ticker = st.selectbox("", ticker_options, label_visibility="collapsed")
    ticker = None if selected_ticker == "All" else selected_ticker

    st.markdown("---")
    st.markdown("**💡 Sample Questions**")
    questions_by_ticker = {
        "NVDA": [
            "What are NVIDIA's main business segments?",
            "What risks does NVIDIA mention in their filing?",
            "What drove NVIDIA's revenue growth in fiscal year 2025?",
        ],
        "AAPL": [
            "How does Apple describe its competition?",
            "What products does Apple sell?",
        ],
        "META": [
            "What are Meta's plans for AI investment?",
            "What regulatory risks does Meta face?",
        ],
        "GOOGL": [
            "How does Alphabet describe its cloud business?",
        ],
        "MSI": [
            "What does Motorola Solutions do?",
            "What are the key risk factors for Motorola Solutions?",
        ],
    }
    if ticker:
        filtered_questions = questions_by_ticker.get(ticker, [])
    else:
        filtered_questions = [q for qs in questions_by_ticker.values() for q in qs]
    sample_questions = ["Select a sample question..."] + filtered_questions
    selected_q = st.selectbox("", sample_questions, label_visibility="collapsed")
    if st.button("Ask this question", use_container_width=True) and selected_q != "Select a sample question...":
        st.session_state["pending_question"] = selected_q

    st.markdown("---")
    st.markdown("**🛠 Stack**")
    st.markdown("- 🔍 **Embeddings:** Nomic AI")
    st.markdown("- 🤖 **LLM:** Groq (llama-3.1-8b)")
    st.markdown("- 🗄 **Vector DB:** Qdrant Cloud")
    st.markdown("- 📄 **Data:** SEC EDGAR 10-K filings")

    st.markdown("---")
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("💡 Select a company from the sidebar or ask about all companies at once.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📄 View Sources"):
                for s in message["sources"]:
                    st.markdown(
                        f'<span class="source-badge">{s["ticker"]}</span>'
                        f'<span class="score-text">relevance: {s["score"]:.2f}</span>',
                        unsafe_allow_html=True
                    )
                    st.caption(s["text"][:200] + "...")
                    st.divider()

chat_input = st.chat_input("Ask a question about the filings...")

if "pending_question" in st.session_state:
    prompt = st.session_state.pop("pending_question")
elif chat_input:
    prompt = chat_input
else:
    prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching filings..."):
            result = answer_question(prompt, ticker=ticker, chat_history=st.session_state.messages)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📄 View Sources"):
                for s in result["sources"]:
                    st.markdown(
                        f'<span class="source-badge">{s["ticker"]}</span>'
                        f'<span class="score-text">relevance: {s["score"]:.2f}</span>',
                        unsafe_allow_html=True
                    )
                    st.caption(s["text"][:200] + "...")
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
