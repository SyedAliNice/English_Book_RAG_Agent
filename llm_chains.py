"""
LLM Chains powered by Groq (llama-3.3-70b-versatile).

Key improvement: The complete, accurate Table of Contents for
"Exploring English Grade 3" is hardcoded here and injected into
every prompt. The LLM now always knows the exact unit titles and
their page numbers, regardless of what OCR extracted.
"""

from typing import List, Dict

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

try:
    from groq import RateLimitError, AuthenticationError, APIError
except ImportError:
    from groq._exceptions import RateLimitError, AuthenticationError, APIError


# ══════════════════════════════════════════════════════════════
# BOOK STRUCTURE — Hardcoded from visual inspection of the PDF
# Unit number → (title, starting page number in the book)
# ══════════════════════════════════════════════════════════════
BOOK_TOC = {
    1:  ("The Drawn Match",              1),
    2:  ("The Joy of Helping Others",   14),
    3:  ("My Village",                  26),
    4:  ("Animals Friends (Poem)",      41),
    5:  ("All are Equal (Article)",     53),
    6:  ("Hazrat Umar (R.A.)",          68),
    7:  ("The Uses of Mobile Phones",   84),
    8:  ("Common Professions in Pakistan", 98),
    9:  ("Keep Our World Clean (Poem)", 113),
    10: ("Staying Safe at Home",        126),
}

# Human-readable string injected into every system prompt
TOC_STRING = "\n".join(
    f"  Unit {num}: {title} (starts at page {page})"
    for num, (title, page) in BOOK_TOC.items()
)

BOOK_CONTEXT = f"""
COMPLETE TABLE OF CONTENTS — Exploring English Grade 3 (Zahid Publications):
{TOC_STRING}

IMPORTANT RULES about this book's structure:
- The book calls each chapter a "Unit", NOT a "Chapter".
- If a student asks for "chapter 2", they mean "Unit 2".
- Each Unit has ONE main title shown in a coloured banner at the start.
- The unit title IS the name of the unit — there is no separate chapter name.
- Always use the exact unit title from the TOC above when answering.
""".strip()


# ── LLM factory ───────────────────────────────────────────────

def get_llm(api_key: str, temperature: float = 0.3) -> ChatGroq:
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=2048,
    )


# ── Helpers ───────────────────────────────────────────────────

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
    try:
        result = chain.invoke(payload)
        return result.content
    except AuthenticationError:
        raise ValueError(
            "❌ **Invalid Groq API key.** Please check the key in the sidebar "
            "(it should start with `gsk_`)."
        )
    except RateLimitError:
        raise ValueError(
            "⏳ **Groq rate limit reached.** Wait 1–2 minutes and try again, "
            "or upgrade your Groq plan at console.groq.com."
        )
    except APIError as e:
        raise ValueError(f"🔴 **Groq API error:** {e}")


# ── 1. General Q&A ────────────────────────────────────────────

QA_SYSTEM = """You are a helpful and encouraging English teacher for Grade 3 students.
You are answering questions about the textbook "Exploring English - Grade 3" by Zahid Publications.

{book_context}

Answer rules:
- For unit/chapter name questions: ALWAYS use the TOC above — you already have all titles.
  Do NOT say "it is not in the excerpts" for unit name questions.
- For content questions: use the provided textbook excerpts.
- Keep answers clear, simple, and age-appropriate.
- If content is not in the excerpts AND not answerable from the TOC, say so honestly.
- Mention page numbers when helpful.
- "Chapter N" and "Unit N" refer to the same thing in this book.
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
            "book_context": BOOK_CONTEXT,
            "context":      format_context(context_docs),
            "question":     question,
        })
    except ValueError as e:
        return _error_dict(str(e))

    sources = list(dict.fromkeys([
        f"Page {d.metadata.get('page','?')} ({d.metadata.get('unit','?')})"
        for d in context_docs
    ]))
    return {"answer": content, "sources": sources}


# ── 2. Exercise Helper ────────────────────────────────────────

EXERCISE_SYSTEM = """You are an expert Grade 3 English teacher solving textbook exercises.
You are working with "Exploring English - Grade 3" by Zahid Publications.

{book_context}

Exercise solving rules:
- Fill-in-the-blank: provide the complete sentence with blanks filled.
- Matching: list each matched pair clearly.
- Grammar: explain the rule briefly before answering.
- Comprehension: answer from the passage and cite the line.
- Always explain WHY the answer is correct so the student learns.
- "Chapter N" and "Unit N" mean the same thing in this book.
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
            "book_context": BOOK_CONTEXT,
            "context":      format_context(context_docs),
            "question":     question,
        })
    except ValueError as e:
        return _error_dict(str(e))

    sources = list(dict.fromkeys([
        f"Page {d.metadata.get('page','?')} ({d.metadata.get('unit','?')})"
        for d in context_docs
    ]))
    return {"answer": content, "sources": sources}


# ── 3. Exam Paper Generator ───────────────────────────────────

EXAM_SYSTEM = """You are an experienced Grade 3 English exam paper setter.
You are creating an exam for "Exploring English - Grade 3" by Zahid Publications.

{book_context}

Formatting rules:
- Output in clean Markdown.
- Include a header: school name placeholder, subject, grade, date, total marks.
- Organise into clearly labelled sections (Q1, Q2, ...).
- Every question must show marks in brackets.
- Base questions on the provided textbook excerpts AND the unit content described in the TOC.
- Do NOT include an answer key in the paper itself.
- End with a "Best of Luck!" footer.
- "Chapter N" and "Unit N" mean the same thing — use "Unit N" consistently.
"""

EXAM_HUMAN = """Textbook content excerpts:
{context}

Exam specification:
- Total marks: {total_marks}
- Difficulty: {difficulty}
- Question types: {q_types}
- Focus topic/unit: {focus}

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
            "book_context": BOOK_CONTEXT,
            "context":      format_context(context_docs),
            "total_marks":  total_marks,
            "difficulty":   difficulty,
            "q_types":      ", ".join(q_types),
            "focus":        focus,
        })
    except ValueError as e:
        return str(e)


# ── 4. Answer Key Generator ───────────────────────────────────

ANSWER_KEY_SYSTEM = """You are an experienced Grade 3 English teacher.
Given an exam paper and the relevant textbook content, produce a detailed answer key.
Format clearly in Markdown. Include the answer AND a brief explanation for each question.

{book_context}
"""

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
            "book_context": BOOK_CONTEXT,
            "context":      format_context(context_docs),
            "exam_paper":   exam_paper,
        })
    except ValueError as e:
        return str(e)
