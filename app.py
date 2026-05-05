"""
Exploring English Grade 3 – RAG Application
Streamlit frontend with three modes:
  1. Ask a Question  (general Q&A)
  2. Exercise Helper (textbook exercises)
  3. Exam Paper Generator

Run with:  streamlit run app.py --server.fileWatcherType none
(The --server.fileWatcherType none flag silences the torchvision warnings
 from Streamlit's file watcher scanning the transformers package.)
"""

import os
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Exploring English Grade 3 – AI Tutor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from rag_engine  import initialise_rag, retrieve_context, retrieve_unit_context, \
                        extract_text_from_pdf, split_documents, build_vector_store, \
                        vector_store_exists
from llm_chains  import get_llm, answer_question, answer_exercise, \
                        detect_unit_summary_request, \
                        generate_exam_paper, generate_answer_key

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem; border-radius: 12px;
        color: white; margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }
    .mode-card {
        background: #f8f9fa; border-left: 4px solid #667eea;
        border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 1rem;
    }
    .answer-box {
        background: #ffffff; border: 1px solid #e0e0e0;
        border-radius: 10px; padding: 1.2rem 1.5rem;
        margin-top: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .source-tag {
        display: inline-block; background: #e8eaf6; color: #3949ab;
        border-radius: 20px; padding: 0.15rem 0.7rem;
        font-size: 0.8rem; margin: 0.2rem 0.15rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
    }
    .stButton > button:hover { opacity: 0.92; }
    .exam-paper {
        font-family: "Georgia", serif; background: #fffef7;
        border: 1px solid #ddd; border-radius: 10px;
        padding: 2rem; margin-top: 1rem; line-height: 1.75;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/book.png", width=70)
    st.markdown("## ⚙️ Settings")

    groq_api_key = st.text_input(
        "🔑 Groq API Key", type="password", placeholder="gsk_…",
        help="Get your free key at console.groq.com",
    )
    pdf_path = st.text_input(
        "📄 PDF Path", value="english_3.pdf",
        help="Path to the textbook PDF (same folder as app.py)",
    )

    st.divider()
    mode = st.radio(
        "📂 Choose Mode",
        ["💬 Ask a Question", "✏️ Exercise Helper", "📝 Exam Paper Generator"],
    )

    st.divider()
    rebuild = st.button("🔄 Rebuild Index")

    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        "This app uses **Groq (LLaMA 3.3-70B)** + **ChromaDB** + **OCR** "
        "to answer questions, solve exercises, and generate exam papers from "
        "_Exploring English Grade 3_."
    )
    st.markdown("**Tip:** Run with `--server.fileWatcherType none` to silence torchvision warnings.")


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📚 Exploring English Grade 3 – AI Tutor</h1>
  <p>Powered by Groq LLaMA 3.3-70B &nbsp;|&nbsp; Zahid Publications</p>
</div>
""", unsafe_allow_html=True)

if not groq_api_key:
    st.info("👈 Please enter your **Groq API key** in the sidebar to get started.")
    st.stop()

if not Path(pdf_path).exists():
    st.error(f"❌ PDF not found at `{pdf_path}`. Place `english_3.pdf` next to `app.py`.")
    st.stop()


# ── RAG Initialisation with OCR progress bar ──────────────────────────────────

def build_index_with_progress(pdf: str):
    """Build the vector index with a real-time OCR progress bar."""
    total_pages_ref = [0]
    progress_bar  = st.progress(0, text="Starting OCR extraction…")
    status_text   = st.empty()

    def on_page(page_num, total):
        total_pages_ref[0] = total
        pct = int((page_num / total) * 100)
        progress_bar.progress(pct, text=f"🔍 OCR: Page {page_num} / {total}")
        status_text.caption(f"Extracting text from page {page_num}…")

    raw_docs = extract_text_from_pdf(pdf, status_callback=on_page)
    progress_bar.progress(100, text="✅ OCR complete — building index…")
    status_text.caption(f"Extracted {len(raw_docs)} pages. Chunking and indexing…")

    chunks = split_documents(raw_docs)
    status_text.caption(f"Created {len(chunks)} chunks. Embedding and storing…")
    vectordb = build_vector_store(chunks)

    progress_bar.empty()
    status_text.empty()
    return vectordb


@st.cache_resource(show_spinner=False)
def load_rag(pdf: str, _rebuild_flag: bool = False):
    if not _rebuild_flag and vector_store_exists():
        with st.spinner("⏳ Loading existing index…"):
            from rag_engine import load_vector_store
            return load_vector_store()
    return build_index_with_progress(pdf)


# Trigger rebuild if button clicked (clear cache first)
if rebuild:
    import shutil
    if Path("chroma_db").exists():
        shutil.rmtree("chroma_db")
    st.cache_resource.clear()

vectordb = load_rag(pdf_path, rebuild)
llm      = get_llm(groq_api_key)

st.success("✅ Textbook indexed and ready!", icon="🎉")
st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1: Ask a Question
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "💬 Ask a Question":
    st.markdown("### 💬 Ask a Question")
    st.markdown(
        '<div class="mode-card">Ask anything about the textbook — units, stories, '
        'grammar rules, vocabulary, or chapter content.</div>',
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Type your question here…")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                # Smart routing: detect summary requests and fetch full unit
                unit_num  = detect_unit_summary_request(question)
                unit_docs = retrieve_unit_context(vectordb, unit_num) if unit_num else []
                docs      = retrieve_context(vectordb, question, k=6)
                result    = answer_question(llm, docs, question, unit_docs=unit_docs or None)

            st.markdown(result["answer"])
            if result["sources"]:
                source_html = " ".join(
                    f'<span class="source-tag">📖 {s}</span>'
                    for s in result["sources"]
                )
                st.markdown(f"**Sources:** {source_html}", unsafe_allow_html=True)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": result["answer"]}
            )

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2: Exercise Helper
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "✏️ Exercise Helper":
    st.markdown("### ✏️ Exercise Helper")
    st.markdown(
        '<div class="mode-card">Paste any exercise from the textbook and get '
        'a detailed answer with explanation.</div>',
        unsafe_allow_html=True,
    )

    exercise_input = st.text_area(
        "📋 Paste or type the exercise question:",
        height=150,
        placeholder=(
            "e.g. Fill in the blanks with the correct form of the verb:\n"
            "1. She ___ (play) cricket every day.\n"
            "2. They ___ (run) in the park.\n"
        ),
    )
    ex_type = st.selectbox(
        "Exercise type (optional)",
        ["Auto-detect", "Fill in the Blanks", "True / False",
         "Multiple Choice (MCQs)", "Comprehension Questions",
         "Matching", "Grammar Exercise", "Vocabulary / Word Meanings", "Writing Exercise"],
    )

    if st.button("✅ Solve Exercise"):
        if not exercise_input.strip():
            st.warning("Please paste an exercise question first.")
        else:
            query = f"[{ex_type}] {exercise_input}" if ex_type != "Auto-detect" else exercise_input
            with st.spinner("Solving…"):
                docs   = retrieve_context(vectordb, query, k=6)
                result = answer_exercise(llm, docs, query)

            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown("#### 📘 Answer & Explanation")
            st.markdown(result["answer"])
            if result["sources"]:
                source_html = " ".join(
                    f'<span class="source-tag">📖 {s}</span>'
                    for s in result["sources"]
                )
                st.markdown(f"**Sources:** {source_html}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 3: Exam Paper Generator
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "📝 Exam Paper Generator":
    st.markdown("### 📝 Exam Paper Generator")
    st.markdown(
        '<div class="mode-card">Configure your exam and the AI will generate '
        'a complete, printable paper based on the textbook content.</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        total_marks = st.slider("📊 Total Marks", 10, 100, 50, step=5)
        difficulty  = st.selectbox("🎯 Difficulty Level", ["Easy", "Medium", "Hard", "Mixed"])
        focus_unit  = st.text_input(
            "🔍 Focus Topic / Unit (leave blank for all)",
            placeholder="e.g. Unit 1: The Drawn Match",
        )
    with col_b:
        q_types = st.multiselect(
            "📋 Question Types",
            ["Multiple Choice Questions (MCQs)", "Fill in the Blanks",
             "True / False", "Short Answer Questions", "Comprehension Passage",
             "Match the Columns", "Write sentences using the given words",
             "Correct the sentences", "Rearrange the words to form sentences"],
            default=["Multiple Choice Questions (MCQs)", "Fill in the Blanks",
                     "True / False", "Short Answer Questions",
                     "Write sentences using the given words"],
        )

    include_answer_key = st.checkbox("🗝️ Also generate Answer Key", value=True)

    if st.button("🚀 Generate Exam Paper"):
        if not q_types:
            st.warning("Please select at least one question type.")
        else:
            search_query = focus_unit if focus_unit else "English grammar vocabulary comprehension Grade 3 unit"
            with st.spinner("📄 Generating exam paper…"):
                docs  = retrieve_context(vectordb, search_query, k=10)
                paper = generate_exam_paper(
                    llm, docs,
                    total_marks=total_marks,
                    difficulty=difficulty,
                    q_types=q_types,
                    focus=focus_unit or "All units",
                )

            st.markdown("---")
            st.markdown("#### 📄 Generated Exam Paper")
            st.markdown(f'<div class="exam-paper">{paper}</div>', unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Exam Paper (.txt)", data=paper,
                file_name="exam_paper_grade3_english.txt", mime="text/plain",
            )

            if include_answer_key:
                st.markdown("---")
                with st.spinner("🗝️ Generating answer key…"):
                    key = generate_answer_key(llm, docs, paper)
                with st.expander("🗝️ View Answer Key (Teacher Only)", expanded=False):
                    st.markdown(key)
                    st.download_button(
                        "⬇️ Download Answer Key (.txt)", data=key,
                        file_name="answer_key_grade3_english.txt",
                        mime="text/plain", key="dl_key",
                    )

            st.success("✅ Exam paper ready!", icon="🎉")
