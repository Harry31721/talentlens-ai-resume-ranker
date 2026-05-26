# 🔍 TalentLens — AI Resume Ranker

> AI-powered resume ranking and candidate intelligence platform built with Streamlit, FAISS, SentenceTransformers, and LangChain.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.x-lightgreen?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=flat-square)

---

## 📌 Overview

TalentLens helps recruiters automatically rank, compare, and analyse candidates by matching their resumes against a job description using semantic similarity. Instead of manually reading every CV, recruiters get an instant ranked list, AI-generated summaries, interview questions, and a conversational assistant — all in one interface.

---

## ✨ Features

- **Semantic Resume Ranking** — Ranks uploaded PDFs against a job description using sentence embeddings and cosine similarity
- **FAISS Vector Search** — Efficiently retrieves the most relevant resumes for any query
- **AI Insights** — Generates a professional candidate summary and 5 tailored interview questions for the top candidate
- **Recruiter AI Agent** — Ask cross-candidate questions like *"Who has the most Machine Learning experience?"* and get grounded answers from all resumes
- **Chat Assistant** — Conversational interface with optional memory for follow-up questions about specific candidates
- **Export** — Download ranked results as a CSV file

---

## 🖥️ Demo

> ⚠️ **Local LLM Notice** — See the important note below before running.

![TalentLens Screenshot](Screenshots/HomePage.png)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| Similarity Ranking | scikit-learn cosine similarity |
| LLM | Mistral via Ollama (local) |
| Conversational Memory | LangChain `InMemoryChatMessageHistory` |
| PDF Parsing | PyPDF2 |

---

## 📁 Project Structure

```
talentlens-ai-resume-ranker/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
└── utils/
    ├── agent.py            # Recruiter AI agent with full resume context
    ├── embeddings.py       # SentenceTransformer embedding generation
    ├── langchain_chat.py   # LangChain chat with session memory
    ├── llm.py              # Resume summarisation, Q-gen, chatbot via Ollama
    ├── parser.py           # PDF text extraction
    ├── ranking.py          # Cosine similarity scoring
    ├── vector_store.py     # FAISS-based resume retrieval
    └── vectors_store.py    # FAISS index creation and search
```

---

## ⚙️ Local Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- Mistral model pulled via Ollama

### 1. Clone the repository

```bash
git clone https://github.com/Harry31721/talentlens-ai-resume-ranker
cd talentlens-ai-resume-ranker
```

### 2. Create and activate a virtual environment

```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the Mistral model via Ollama

```bash
ollama pull mistral
```

### 5. Start Ollama (if not already running)

```bash
ollama serve
```

### 6. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🚀 How to Use

1. **Enter a Job Description** in the text area on the left
2. **Upload Resume PDFs** using the file uploader
3. Click **Rank Candidates** — the app embeds and scores all resumes
4. Explore the results across four tabs:
   - `📊 Rankings` — sorted candidate table and score chart
   - `🤖 AI Insights` — summary and interview questions for the top candidate
   - `🧠 AI Agent` — ask questions across all candidates at once
   - `💬 Chat Assistant` — follow-up questions with optional conversation memory

---

## ⚠️ Important: Local-Only LLM (Ollama)

This project uses **Ollama running locally** to serve the Mistral language model. This means:

- ✅ Works perfectly on your **local machine** after following the setup steps above
- ❌ **Will not work** on Streamlit Cloud or any remote deployment as-is, because those servers cannot connect to your local Ollama instance
- The **ranking, FAISS search, and PDF parsing** features work on any deployment since they don't depend on Ollama
- Only the **AI Insights, Agent, and Chat** features require the local LLM

### Want to deploy fully to Streamlit Cloud?

Replace Ollama with a cloud LLM API:
- [OpenAI API](https://platform.openai.com) — swap `ChatOllama` for `ChatOpenAI`
- [Anthropic Claude API](https://console.anthropic.com) — swap for `ChatAnthropic`

Store your API key as a [Streamlit Secret](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) and you're good to go.

---

## 📦 Requirements

```
streamlit
pandas
PyPDF2
sentence-transformers
faiss-cpu
scikit-learn
langchain
langchain-ollama
langchain-core
ollama
numpy
```

---

## 🗺️ Roadmap

- [ ] Swap Ollama for OpenAI / Claude API for full cloud deployment
- [ ] Add support for DOCX resumes
- [ ] Candidate comparison view (side-by-side)
- [ ] Persistent vector store across sessions
- [ ] Export AI insights as a PDF report

---

## 👤 Author

**Harikrishan Chauhan**
[LinkedIn](www.linkedin.com/in/harikrishan-chauhan) · [GitHub](https://github.com/Harry31721)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).