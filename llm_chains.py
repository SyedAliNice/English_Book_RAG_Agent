"""
LLM Chains powered by Groq (llama-3.3-70b-versatile).

Changes vs original:
  - Graceful error handling for RateLimitError, AuthenticationError, APIError
  - answer_question now returns a fallback dict on any LLM error
  - Prompt improved: LLM is told to infer unit names from context clues
    (headings, chapter markers) even when the word "Unit N" is absent.
"""

from typing import List, Dict

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

try:
    from groq import RateLimitError, AuthenticationError, APIError
except ImportError:          # older groq versions use a different path
    from groq._exceptions import RateLimitError, AuthenticationError, APIError


# ── LLM factory ──────────────────────────────────────────────────────────────

def get_llm(api_key: str, temperature: float = 0.3) -> ChatGroq:
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=2048,
    )


# ── Helper ───────────────────────────────────────────────────────────────────

def format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        parts.append(
            f"[Excerpt {i} | Page {meta.get('page','?')} | {meta.get('unit','?')}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def _error_dict(msg: str) -> Dict:
    return {"answer": msg, "sources": []}


def _invoke_safe(chain, payload: dict) -> str:
    """Invoke a LangChain chain and surface friendly errors on failure."""
    try:
        result = chain.invoke(payload)
        return result.content
    except AuthenticationError:
        raise ValueError(
            "❌ **Invalid Groq API key.** Please check the key you entered in the sidebar "
            "and make sure it starts with `gsk_`."
        )
    except RateLimitError:
        raise ValueError(
            "⏳ **Groq rate limit reached.** You've hit the free-tier limit. "
            "Wait 1–2 minutes and try again, or upgrade your Groq plan."
        )
    except APIError as e:
        raise ValueError(f"🔴 **Groq API error:** {e}")


# ── 1. General Q&A ───────────────────────────────────────────────────────────

QA_SYSTEM = """You are a helpful and encouraging English teacher for Grade 3 students.
Answer questions based ONLY on the provided excerpts from the textbook
"Exploring English - Grade 3" by Zahid Publications.

Rules:
- Keep answers clear, simple, and age-appropriate.
- The excerpts are OCR-scanned, so headings and unit titles may appear as plain text
  lines (e.g. "The Drawn Match", "A New Friend") — treat these as unit/chapter names.
- If asked about a unit name or chapter title, look for bold headings, numbered
  headings, or prominent capitalized phrases in the excerpts and report those.
- If the answer is truly not in the excerpts, say so honestly.
- Quote the relevant part of the text when helpful.
- Mention the page number if you can infer it from the excerpts.
"""

QA_HUMAN = """Textbook excerpts:
{context}

Student's question: {question}

Please answer clearly and helpfully."""


def answer_question(llm: ChatGroq, context_docs: List[Document], question: str) -> Dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM),
        ("human",  QA_HUMAN),
    ])
    chain = prompt | llm
    try:
        content = _invoke_safe(chain, {
            "context":  format_context(context_docs),
            "question": question,
        })
    except ValueError as e:
        return _error_dict(str(e))

    sources = list(dict.fromkeys([
        f"Page {d.metadata.get('page','?')} ({d.metadata.get('unit','?')})"
        for d in context_docs
    ]))
    return {"answer": content, "sources": sources}


# ── 2. Exercise Q&A ──────────────────────────────────────────────────────────

EXERCISE_SYSTEM = """You are an expert Grade 3 English teacher solving textbook exercises.
Use the provided excerpts to answer the exercise question accurately.

Rules:
- Fill-in-the-blank: provide the complete sentence with blanks filled.
- Matching: list each matched pair clearly.
- Grammar: explain the rule briefly before answering.
- Comprehension: answer from the passage and cite the line.
- Always explain WHY the answer is correct so the student learns.
"""

EXERCISE_HUMAN = """Textbook excerpts:
{context}

Exercise question:
{question}

Provide the answer with a brief explanation."""


def answer_exercise(llm: ChatGroq, context_docs: List[Document], question: str) -> Dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXERCISE_SYSTEM),
        ("human",  EXERCISE_HUMAN),
    ])
    chain = prompt | llm
    try:
        content = _invoke_safe(chain, {
            "context":  format_context(context_docs),
            "question": question,
        })
    except ValueError as e:
        return _error_dict(str(e))

    sources = list(dict.fromkeys([
        f"Page {d.metadata.get('page','?')} ({d.metadata.get('unit','?')})"
        for d in context_docs
    ]))
    return {"answer": content, "sources": sources}


# ── 3. Exam Paper Generation ─────────────────────────────────────────────────

EXAM_SYSTEM = """You are an experienced Grade 3 English exam paper setter.
Create a well-structured exam paper from the provided textbook content.

Formatting rules:
- Output in clean Markdown.
- Include a header with school name placeholder, subject, grade, date, and total marks.
- Organise questions into clearly labelled sections (Q1, Q2, ...).
- Every question must have marks allocated (shown in brackets).
- Questions must be directly based on the provided textbook excerpts.
- Do NOT include an answer key in the exam paper itself.
- End with a "Best of Luck!" footer.
"""

EXAM_HUMAN = """Textbook content to base the exam on:
{context}

Exam specification:
- Total marks: {total_marks}
- Difficulty: {difficulty}
- Question types to include: {q_types}
- Focus topic/unit (if any): {focus}

Generate the complete exam paper now."""


def generate_exam_paper(
    llm: ChatGroq,
    context_docs: List[Document],
    total_marks: int   = 50,
    difficulty: str    = "Medium",
    q_types: List[str] = None,
    focus: str         = "All units",
) -> str:
    if q_types is None:
        q_types = [
            "Multiple Choice Questions (MCQs)",
            "Fill in the Blanks",
            "True / False",
            "Short Answer Questions",
            "Write sentences using the given words",
        ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXAM_SYSTEM),
        ("human",  EXAM_HUMAN),
    ])
    chain = prompt | llm
    try:
        return _invoke_safe(chain, {
            "context":     format_context(context_docs),
            "total_marks": total_marks,
            "difficulty":  difficulty,
            "q_types":     ", ".join(q_types),
            "focus":       focus,
        })
    except ValueError as e:
        return str(e)


# ── 4. Answer Key Generation ─────────────────────────────────────────────────

ANSWER_KEY_SYSTEM = """You are an experienced Grade 3 English teacher.
Given an exam paper and the relevant textbook content, produce a detailed answer key.
Format it clearly in Markdown with each question's answer and a brief explanation."""

ANSWER_KEY_HUMAN = """Textbook content:
{context}

Exam paper:
{exam_paper}

Produce a complete answer key."""


def generate_answer_key(
    llm: ChatGroq,
    context_docs: List[Document],
    exam_paper: str,
) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_KEY_SYSTEM),
        ("human",  ANSWER_KEY_HUMAN),
    ])
    chain = prompt | llm
    try:
        return _invoke_safe(chain, {
            "context":    format_context(context_docs),
            "exam_paper": exam_paper,
        })
    except ValueError as e:
        return str(e)