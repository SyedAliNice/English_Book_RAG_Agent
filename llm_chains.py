"""
LLM Chains powered by Groq (llama-3.3-70b-versatile).

Key improvements in this version:
  1. Hardcoded TOC injected into every prompt — unit names always correct.
  2. Smart query routing:
       - Summary / "tell me about unit X" → retrieves ALL chunks from that
         unit using retrieve_unit_context(), not just 6 random chunks.
       - Normal Q&A → standard MMR retrieval as before.
  3. Graceful error handling for rate limits / bad API keys.
"""

import re
from typing import List, Dict, Optional, Tuple

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

try:
    from groq import RateLimitError, AuthenticationError, APIError
except ImportError:
    from groq._exceptions import RateLimitError, AuthenticationError, APIError


# ══════════════════════════════════════════════════════════════
# BOOK STRUCTURE — confirmed by visual inspection of the PDF
# ══════════════════════════════════════════════════════════════
BOOK_TOC = {
    1:  ("The Drawn Match",                  1),
    2:  ("The Joy of Helping Others",       14),
    3:  ("My Village",                      26),
    4:  ("Animals Friends (Poem)",          41),
    5:  ("All are Equal (Article)",         53),
    6:  ("Hazrat Umar (R.A.)",              68),
    7:  ("The Uses of Mobile Phones",       84),
    8:  ("Common Professions in Pakistan",  98),
    9:  ("Keep Our World Clean (Poem)",    113),
    10: ("Staying Safe at Home",           126),
}

TOC_STRING = "\n".join(
    f"  Unit {num}: {title}  (starts at page {page})"
    for num, (title, page) in BOOK_TOC.items()
)

BOOK_CONTEXT = f"""
COMPLETE TABLE OF CONTENTS — Exploring English Grade 3 (Zahid Publications):
{TOC_STRING}

RULES about this book:
- Each section is called a "Unit", NOT a "Chapter". "Chapter N" = "Unit N".
- Each Unit has exactly ONE title (shown above). Use that exact title always.
- For unit/chapter NAME questions, use the TOC — never say "not in excerpts".
""".strip()


# ══════════════════════════════════════════════════════════════
# SMART QUERY ROUTER
# Detects whether the question is asking for a full unit
# summary/overview and which unit number it refers to.
# ══════════════════════════════════════════════════════════════

# Patterns that signal "give me the full content of a unit"
_SUMMARY_TRIGGERS = re.compile(
    r"\b(summar|overview|about|explain|describe|what (is|was|happen|are)|"
    r"tell me|brief|detail|content|story|main (idea|point|topic))\b",
    re.IGNORECASE,
)

# Patterns to extract a unit/chapter number from the question
_UNIT_NUMBER_PATTERNS = [
    re.compile(r"\b(?:unit|chapter|ch\.?)\s*(\d+)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:unit|chapter|ch\.?)\s+"
        r"(one|two|three|four|five|six|seven|eight|nine|ten)\b",
        re.IGNORECASE,
    ),
]
_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Also match unit titles directly in the question
_TITLE_TO_UNIT = {
    title.lower(): num for num, (title, _) in BOOK_TOC.items()
}


def detect_unit_summary_request(question: str) -> Optional[int]:
    """
    Returns the unit number if the question is asking for a full
    summary / content of a specific unit. Returns None otherwise.
    """
    q = question.lower()

    # Must have at least one summary-trigger word
    if not _SUMMARY_TRIGGERS.search(q):
        return None

    # Try to find a unit number
    for pat in _UNIT_NUMBER_PATTERNS:
        m = pat.search(question)
        if m:
            raw = m.group(1)
            if raw.isdigit():
                n = int(raw)
            else:
                n = _WORD_TO_NUM.get(raw.lower(), 0)
            if 1 <= n <= 10:
                return n

    # Try matching a unit title mentioned in the question
    for title_lower, num in _TITLE_TO_UNIT.items():
        if title_lower in q:
            return num

    return None


# ══════════════════════════════════════════════════════════════
# LLM FACTORY
# ══════════════════════════════════════════════════════════════

def get_llm(api_key: str, temperature: float = 0.3) -> ChatGroq:
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=2048,
    )


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def format_context(docs: List[Document]) -> str:
    if not docs:
        return "(No excerpts retrieved)"
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
        return chain.invoke(payload).content
    except AuthenticationError:
        raise ValueError(
            "❌ **Invalid Groq API key.** Check the key in the sidebar (starts with `gsk_`)."
        )
    except RateLimitError:
        raise ValueError(
            "⏳ **Groq rate limit reached.** Wait 1–2 minutes then try again."
        )
    except APIError as e:
        raise ValueError(f"🔴 **Groq API error:** {e}")


def _make_sources(docs: List[Document]) -> List[str]:
    return list(dict.fromkeys([
        f"Page {d.metadata.get('page','?')} ({d.metadata.get('unit','?')})"
        for d in docs
    ]))


# ══════════════════════════════════════════════════════════════
# 1. GENERAL Q&A  (with smart summary routing)
# ══════════════════════════════════════════════════════════════

QA_SYSTEM = """You are a helpful and encouraging English teacher for Grade 3 students.
You are answering questions about "Exploring English - Grade 3" by Zahid Publications.

{book_context}

Answer rules:
- For unit/chapter NAME questions: use the TOC above — never say "not in excerpts".
- For summary / "what is unit X about" questions: a full set of excerpts from
  that unit has been provided — write a proper, detailed summary from them.
- For other content questions: use the provided excerpts.
- Keep answers clear, simple, and age-appropriate.
- Mention the page number when helpful.
- "Chapter N" and "Unit N" are the same thing in this book.
"""

QA_HUMAN = """Textbook excerpts:
{context}

Student's question: {question}

Please answer clearly and helpfully."""

SUMMARY_SYSTEM = """You are a helpful Grade 3 English teacher.
You have been given ALL the text content from a specific unit of
"Exploring English - Grade 3" by Zahid Publications.

{book_context}

Your task: write a clear, well-structured summary of the unit.
Include:
  • The unit title and number
  • The main story / topic of the unit
  • Key characters (if any) and what happens to them
  • Important grammar or vocabulary topics covered
  • The moral or lesson of the unit (if applicable)

Keep the language simple and suitable for Grade 3 students and their parents.
Do NOT say the excerpts are missing or incomplete — you have the full unit.
"""

SUMMARY_HUMAN = """Full content of the unit (all pages, in order):
{context}

Question: {question}

Write a complete summary based on the content above."""


def answer_question(
    llm: ChatGroq,
    context_docs: List[Document],
    question: str,
    unit_docs: Optional[List[Document]] = None,
) -> Dict:
    """
    unit_docs: if provided (non-empty), the question is a unit-summary
    request and we use the SUMMARY prompt with the full unit content.
    Otherwise we use the normal QA prompt.
    """
    if unit_docs:
        # ── SUMMARY path ─────────────────────────────────────
        prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARY_SYSTEM),
            ("human",  SUMMARY_HUMAN),
        ])
        docs_to_use = unit_docs
    else:
        # ── Normal Q&A path ───────────────────────────────────
        prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM),
            ("human",  QA_HUMAN),
        ])
        docs_to_use = context_docs

    chain = prompt | llm
    try:
        content = _invoke_safe(chain, {
            "book_context": BOOK_CONTEXT,
            "context":      format_context(docs_to_use),
            "question":     question,
        })
    except ValueError as e:
        return _error_dict(str(e))

    return {"answer": content, "sources": _make_sources(docs_to_use)}


# ══════════════════════════════════════════════════════════════
# 2. EXERCISE HELPER
# ══════════════════════════════════════════════════════════════

EXERCISE_SYSTEM = """You are an expert Grade 3 English teacher solving textbook exercises.
You are working with "Exploring English - Grade 3" by Zahid Publications.

{book_context}

Rules:
- Fill-in-the-blank: give the complete sentence with blanks filled.
- Matching: list each matched pair clearly.
- Grammar: explain the rule briefly before answering.
- Comprehension: answer from the passage and cite the relevant line.
- Always explain WHY the answer is correct.
- "Chapter N" and "Unit N" are the same in this book.
"""

EXERCISE_HUMAN = """Textbook excerpts:
{context}

Exercise:
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
    return {"answer": content, "sources": _make_sources(context_docs)}


# ══════════════════════════════════════════════════════════════
# 3. EXAM PAPER GENERATOR
# ══════════════════════════════════════════════════════════════

EXAM_SYSTEM = """You are an experienced Grade 3 English exam paper setter.
You are creating an exam for "Exploring English - Grade 3" by Zahid Publications.

{book_context}

Formatting rules:
- Output in clean Markdown.
- Header: school name placeholder, subject, grade, date, total marks.
- Sections labelled Q1, Q2, … with marks in brackets.
- Base questions directly on the provided excerpts and the unit TOC.
- Do NOT include an answer key in the paper.
- End with "Best of Luck!".
- Use "Unit N" consistently (not "Chapter N").
"""

EXAM_HUMAN = """Textbook excerpts:
{context}

Exam spec:
- Total marks: {total_marks}
- Difficulty: {difficulty}
- Question types: {q_types}
- Focus: {focus}

Generate the complete exam paper."""


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
            "Multiple Choice Questions (MCQs)", "Fill in the Blanks",
            "True / False", "Short Answer Questions",
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


# ══════════════════════════════════════════════════════════════
# 4. ANSWER KEY GENERATOR
# ══════════════════════════════════════════════════════════════

ANSWER_KEY_SYSTEM = """You are an experienced Grade 3 English teacher.
Produce a detailed answer key for the given exam paper.
Format in Markdown — answer + brief explanation per question.

{book_context}
"""

ANSWER_KEY_HUMAN = """Textbook content:
{context}

Exam paper:
{exam_paper}

Produce the complete answer key."""


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
