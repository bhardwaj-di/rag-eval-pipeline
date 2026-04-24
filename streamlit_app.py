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
    .main { padding-top: 1rem; }
    .stChatMessage { border-radius: 12px; margin-bottom: 0.5rem; }
    .source-badge {
        display: inline-block;
        background: #f0f2f6;
        border-radius: 8px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        color: #1f77b4;
    }
    .score-text { color: #888; font-size: 12px; }
</style>
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
            "What is NVIDIA's growth strategy?",
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
    st.markdown("## 📈 SEC 10-K Filings Assistant")
    st.markdown("Ask questions about annual filings from **AAPL, GOOGL, META, MSI,** and **NVDA**.")
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
