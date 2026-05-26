import streamlit as st
import pandas as pd

from utils.parser import extract_text_from_pdf
from utils.embeddings import generate_embedding, generate_embeddings
from utils.ranking import rank_resumes
from utils.llm import summarize_resume, generate_interview_questions, ask_resume_chatbot
from utils.vector_store import retrieve_relevant_resumes
from utils.vectors_store import create_faiss_index
from utils.agent import ask_agent
from utils.langchain_chat import chat_with_memory, clear_session

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentLens · AI Resume Ranker", page_icon="🔍", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; }

/* ── App header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 2rem 0;
    border-bottom: 1px solid #1e2235;
    margin-bottom: 2rem;
}
.app-header .logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-header .tagline {
    font-size: 0.85rem;
    color: #64748b;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #60a5fa;
    margin-bottom: 0.4rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 1.2rem;
    letter-spacing: -0.2px;
}

/* ── Cards ── */
.card {
    background: #131625;
    border: 1px solid #1e2640;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 3px solid #60a5fa;
}

/* ── Top candidate banner ── */
.top-candidate-banner {
    background: linear-gradient(135deg, #0f1f3d 0%, #131625 100%);
    border: 1px solid #1e3a5f;
    border-left: 4px solid #60a5fa;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
}
.top-candidate-banner .tc-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
}
.top-candidate-banner .tc-label {
    font-size: 0.72rem;
    color: #60a5fa;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
}
.top-candidate-banner .tc-score {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #60a5fa;
}

/* ── Stat cards ── */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.stat-card {
    flex: 1;
    background: #131625;
    border: 1px solid #1e2640;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-card .stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #e2e8f0;
}
.stat-card .stat-label {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 400;
    margin-top: 2px;
}

/* ── Insight box ── */
.insight-box {
    background: #0d1a2d;
    border: 1px solid #1a3050;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    color: #cbd5e1;
    font-size: 0.9rem;
    line-height: 1.7;
    margin-bottom: 1rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.2s ease, transform 0.1s ease !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.25) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* Download button – subtle variant */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid #2d3f5e !important;
    color: #94a3b8 !important;
    box-shadow: none !important;
    font-size: 0.82rem !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #60a5fa !important;
    color: #60a5fa !important;
    transform: none !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0d0f1a;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1e2235;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    padding: 0.45rem 1.1rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: #1e2640 !important;
    color: #e2e8f0 !important;
}

/* ── Text inputs & textareas ── */
.stTextInput input, .stTextArea textarea {
    background: #131625 !important;
    border: 1px solid #1e2640 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #131625 !important;
    border: 1px dashed #1e2640 !important;
    border-radius: 10px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2640 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0f1a !important;
    border-right: 1px solid #1e2235 !important;
}
[data-testid="stSidebar"] .sidebar-brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #60a5fa;
    padding: 0 0 1.2rem 0;
    border-bottom: 1px solid #1e2235;
    margin-bottom: 1.2rem;
}
[data-testid="stSidebar"] label {
    font-size: 0.82rem !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
}
.stSlider [data-baseweb="slider"] { padding: 0 !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #131625 !important;
    border: 1px solid #1e2640 !important;
    border-radius: 12px !important;
    padding: 0.9rem 1.1rem !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stChatInput"] textarea {
    background: #131625 !important;
    border: 1px solid #1e2640 !important;
    border-radius: 10px !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3b82f6 !important; }

/* ── Alerts / warnings ── */
.stAlert {
    border-radius: 8px !important;
    font-size: 0.87rem !important;
}

/* ── Divider ── */
hr { border-color: #1e2235 !important; margin: 2rem 0 !important; }

/* ── Radio ── */
.stRadio label { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ── App Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <div class="logo">🔍 TalentLens</div>
        <div class="tagline">AI-powered resume ranking & candidate intelligence</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚙ Controls</div>', unsafe_allow_html=True)

    st.markdown("**Filter & Search**")
    search_candidate = st.text_input("Search by name", placeholder="e.g. John")
    minimum_score    = st.slider("Minimum Match Score", 0, 100, 50)

    st.markdown("---")
    st.markdown("**Chat Settings**")
    chat_mode = st.radio("Mode", ["Single Turn", "With Memory"], index=1)
    if st.button("Clear Chat History"):
        clear_session("recruiter_session")
        st.session_state.pop("chat_history", None)
        st.success("Cleared.")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#334155; line-height:1.6;">
        Built with Streamlit · FAISS<br>SentenceTransformers · LangChain<br>Ollama · Mistral
    </div>
    """, unsafe_allow_html=True)

# ── Input Section ──────────────────────────────────────────────────────────────
col_jd, col_up = st.columns([1.1, 1], gap="large")

with col_jd:
    st.markdown('<div class="section-label">Step 1</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        label="job_desc",
        label_visibility="collapsed",
        placeholder="Paste the job description here — skills, responsibilities, requirements...",
        height=200
    )

with col_up:
    st.markdown('<div class="section-label">Step 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Resumes</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        label="resumes",
        label_visibility="collapsed",
        accept_multiple_files=True,
        type=["pdf"]
    )
    if uploaded_files:
        st.markdown(f"""
        <div style="font-size:0.8rem; color:#64748b; margin-top:0.4rem;">
            {len(uploaded_files)} file{'s' if len(uploaded_files) > 1 else ''} ready
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section-label" style="margin-top:0.5rem;">Step 3</div>', unsafe_allow_html=True)
rank_btn = st.button("⚡  Rank Candidates", use_container_width=False)

# ── Rank Logic ─────────────────────────────────────────────────────────────────
if rank_btn:
    if not uploaded_files:
        st.warning("Upload at least one resume to continue.")
    elif not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        resume_data = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            if text:
                resume_data.append({"name": file.name, "text": text})

        with st.spinner("Embedding & ranking candidates..."):
            job_embedding    = generate_embedding(job_description)
            resume_texts     = [r["text"] for r in resume_data]
            resume_embeddings = generate_embeddings(resume_texts)
            faiss_index      = create_faiss_index(resume_embeddings)
            scores           = rank_resumes(resume_embeddings, job_embedding)

        results = [
            {"Candidate": resume_data[i]["name"], "Match Score": round(score * 100, 2)}
            for i, score in enumerate(scores)
        ]
        results_df = pd.DataFrame(results)
        results_df = results_df[results_df["Match Score"] >= minimum_score]
        if search_candidate:
            results_df = results_df[results_df["Candidate"].str.contains(search_candidate, case=False)]
        results_df = results_df.sort_values(by="Match Score", ascending=False).reset_index(drop=True)

        st.session_state["results_df"]  = results_df
        st.session_state["resume_data"] = resume_data
        st.session_state["faiss_index"] = faiss_index
        st.session_state.pop("summary",    None)
        st.session_state.pop("questions",  None)

# ── Results ────────────────────────────────────────────────────────────────────
if "results_df" in st.session_state:
    results_df  = st.session_state["results_df"]
    resume_data = st.session_state["resume_data"]
    faiss_index = st.session_state["faiss_index"]

    if results_df.empty:
        st.warning("No candidates meet the minimum match score. Try lowering the threshold.")
    else:
        top = results_df.iloc[0]

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Top Candidate Banner ─────────────────────────────────────────────
        st.markdown(f"""
        <div class="top-candidate-banner">
            <div>
                <div class="tc-label">🏆 Top Match</div>
                <div class="tc-name">{top['Candidate'].replace('.pdf','')}</div>
            </div>
            <div style="text-align:right">
                <div class="tc-score">{top['Match Score']}%</div>
                <div class="tc-label">match score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Stat Row ─────────────────────────────────────────────────────────
        avg  = results_df["Match Score"].mean()
        high = results_df["Match Score"].max()
        low  = results_df["Match Score"].min()
        cnt  = len(results_df)

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-value">{cnt}</div>
                <div class="stat-label">Candidates Matched</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg:.1f}%</div>
                <div class="stat-label">Average Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{high:.1f}%</div>
                <div class="stat-label">Highest Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{low:.1f}%</div>
                <div class="stat-label">Lowest Score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Tabs ─────────────────────────────────────────────────────────────
        tab_rank, tab_insights, tab_agent, tab_chat = st.tabs([
            "📊  Rankings",
            "🤖  AI Insights",
            "🧠  AI Agent",
            "💬  Chat Assistant"
        ])

        # ── Tab 1 · Rankings ─────────────────────────────────────────────────
        with tab_rank:
            col_tbl, col_chart = st.columns([1, 1.1], gap="large")
            with col_tbl:
                st.markdown('<div class="section-title">Candidate Rankings</div>', unsafe_allow_html=True)
                st.dataframe(
                    results_df.style.background_gradient(subset=["Match Score"], cmap="Blues"),
                    use_container_width=True, hide_index=True
                )
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "↓  Export CSV", data=csv,
                    file_name="talent_lens_rankings.csv", mime="text/csv"
                )
            with col_chart:
                st.markdown('<div class="section-title">Score Distribution</div>', unsafe_allow_html=True)
                st.bar_chart(
                    results_df.set_index("Candidate")["Match Score"],
                    use_container_width=True, height=320
                )

        # ── Tab 2 · AI Insights ──────────────────────────────────────────────
        with tab_insights:
            st.markdown('<div class="section-title">AI Candidate Insights</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:0.85rem; color:#64748b; margin-bottom:1.2rem;">
                Generating insights for <strong style="color:#94a3b8">{top['Candidate'].replace('.pdf','')}</strong>
                — the top-ranked candidate.
            </div>
            """, unsafe_allow_html=True)

            if st.button("Generate Insights & Interview Questions"):
                top_text = next(r["text"] for r in resume_data if r["name"] == top["Candidate"])
                with st.spinner("Analyzing resume..."):
                    summary   = summarize_resume(top_text)
                    questions = generate_interview_questions(top_text)
                st.session_state["summary"]   = summary
                st.session_state["questions"] = questions

            if "summary" in st.session_state:
                col_s, col_q = st.columns(2, gap="large")
                with col_s:
                    st.markdown('<div class="section-label">Candidate Summary</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="insight-box">{st.session_state["summary"]}</div>',
                                unsafe_allow_html=True)
                with col_q:
                    st.markdown('<div class="section-label">Suggested Interview Questions</div>',
                                unsafe_allow_html=True)
                    st.markdown(f'<div class="insight-box">{st.session_state["questions"]}</div>',
                                unsafe_allow_html=True)

        # ── Tab 3 · AI Agent ─────────────────────────────────────────────────
        with tab_agent:
            st.markdown('<div class="section-title">Recruiter AI Agent</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.85rem; color:#64748b; margin-bottom:1.4rem;">
                The agent reads <strong style="color:#94a3b8">all uploaded resumes</strong> and
                answers cross-candidate questions with full context.
            </div>
            """, unsafe_allow_html=True)

            agent_query = st.text_area(
                label="agent_q",
                label_visibility="collapsed",
                placeholder=(
                    "Try asking:\n"
                    "• Who has the most years of Machine Learning experience?\n"
                    "• Which candidate has led engineering teams?\n"
                    "• Compare the top 2 candidates for a senior backend role"
                ),
                height=130
            )
            if st.button("Run Agent  →"):
                if not agent_query.strip():
                    st.warning("Please type a question first.")
                else:
                    with st.spinner("Reasoning across all candidates..."):
                        context = "\n\n".join(
                            [f"Candidate: {r['name']}\n{r['text']}" for r in resume_data]
                        )
                        response = ask_agent(agent_query, resume_context=context)
                    st.markdown('<div class="section-label" style="margin-top:1rem;">Agent Response</div>',
                                unsafe_allow_html=True)
                    st.markdown(f'<div class="insight-box card-accent">{response}</div>',
                                unsafe_allow_html=True)

        # ── Tab 4 · Chat Assistant ────────────────────────────────────────────
        with tab_chat:
            st.markdown('<div class="section-title">Chat Assistant</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:0.85rem; color:#64748b; margin-bottom:1.2rem;">
                Mode: <strong style="color:#94a3b8">{chat_mode}</strong>
                &nbsp;·&nbsp; Retrieves the top-3 most relevant resumes per query via FAISS.
            </div>
            """, unsafe_allow_html=True)

            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []

            chat_container = st.container()
            with chat_container:
                for msg in st.session_state["chat_history"]:
                    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                        st.write(msg["content"])

            recruiter_query = st.chat_input("Ask about any candidate...")
            if recruiter_query:
                with chat_container:
                    with st.chat_message("user"):
                        st.write(recruiter_query)
                st.session_state["chat_history"].append({"role": "user", "content": recruiter_query})

                with st.spinner("Retrieving context & generating answer..."):
                    relevant = retrieve_relevant_resumes(
                        recruiter_query, generate_embedding, faiss_index, resume_data, top_k=3
                    )
                    context = "\n\n".join([f"[{r['name']}]\n{r['text']}" for r in relevant])

                    if chat_mode == "With Memory":
                        prompt = (
                            f"You are a recruiter assistant. Using ONLY the resume data below, "
                            f"give a detailed and specific answer. Always refer to candidates by name.\n\n"
                            f"RESUME DATA:\n{context}\n\n"
                            f"QUESTION: {recruiter_query}\n\nANSWER:"
                        )
                        response = chat_with_memory(prompt, session_id="recruiter_session")
                    else:
                        response = ask_resume_chatbot(recruiter_query, context)

                with chat_container:
                    with st.chat_message("assistant"):
                        st.write(response)
                st.session_state["chat_history"].append({"role": "assistant", "content": response})