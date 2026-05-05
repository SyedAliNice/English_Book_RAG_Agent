# 📚 Exploring English Grade 3 – RAG AI Tutor

A Retrieval-Augmented Generation (RAG) application built on the
**Exploring English Grade 3** textbook (Zahid Publications).
Powered by **Groq (LLaMA 3.3-70B)** + **ChromaDB** + **Streamlit**.

---

## 🗂️ Project Structure

```
english_rag_app/
├── app.py              ← Streamlit frontend (run this)
├── rag_engine.py       ← PDF ingestion + vector store (ChromaDB)
├── llm_chains.py       ← Groq LLM chains for Q&A, exercises, exams
├── requirements.txt    ← Python dependencies
├── english_3.pdf       ← ← ← Place your PDF here!
└── chroma_db/          ← Auto-created vector database
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run downloads the embedding model (~90 MB).

### 2. Get a Groq API Key (free)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up / log in
3. Create a new API key
4. Copy it — you'll paste it in the app sidebar

### 3. Place the PDF

Copy `english_3.pdf` into the `english_rag_app/` folder (same folder as `app.py`).

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🎯 Features

### 💬 Ask a Question
- General questions about the book (grammar rules, vocabulary, story summaries)
- Persistent chat history in the session
- Source page references shown for every answer

### ✏️ Exercise Helper
- Paste any exercise from the textbook
- Select the exercise type (fill-in-the-blanks, MCQ, comprehension, etc.)
- Get a detailed answer **with explanation**

### 📝 Exam Paper Generator
- Configure total marks (10–100), difficulty, and question types
- Focus on a specific unit or generate from all content
- Download the exam paper as a `.txt` file
- Optionally generate an **Answer Key** (teacher view)

---

## 🔧 Configuration

| Setting | Default | Notes |
|---------|---------|-------|
| Embedding model | `all-MiniLM-L6-v2` | Free, runs on CPU |
| LLM | `llama-3.3-70b-versatile` | Via Groq free tier |
| Chunk size | 600 tokens | Tuned for textbook pages |
| Retrieval k | 5–10 docs | Adjustable in `rag_engine.py` |

To switch the LLM model, edit `llm_chains.py → get_llm()`.

---

## 🛠️ Rebuilding the Index

If you replace the PDF or want a fresh index:

- Click **🔄 Rebuild Index** in the sidebar, **OR**
- Delete the `chroma_db/` folder and restart the app.

---

## 📦 Dependencies

- `streamlit` – UI
- `langchain` + `langchain-groq` – LLM orchestration
- `langchain-community` – ChromaDB integration
- `chromadb` – local vector database
- `sentence-transformers` – free embeddings (`all-MiniLM-L6-v2`)
- `pymupdf` – fast PDF text extraction

---

## 💡 Tips

- The **first launch** takes ~1–2 minutes to index all 142 pages.
- Subsequent launches load the cached index in seconds.
- For best exam papers, specify the unit name in the **Focus Topic** field.
- The answer key is hidden behind an expander so students can't see it.
