# """
# LLM Chains powered by Groq (llama-3.3-70b-versatile).

# Changes vs original:
#   - Graceful error handling for RateLimitError, AuthenticationError, APIError
#   - answer_question now returns a fallback dict on any LLM error
#   - Prompt improved: LLM is told to infer unit names from context clues
#     (headings, chapter markers) even when the word "Unit N" is absent.
# """

# from typing import List, Dict

# from langchain_groq import ChatGroq
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate

# try:
#     from groq import RateLimitError, AuthenticationError, APIError
# except ImportError:          # older groq versions use a different path
#     from groq._exceptions import RateLimitError, AuthenticationError, APIError


# # ── LLM factory ──────────────────────────────────────────────────────────────

# def get_llm(api_key: str, temperature: float = 0.3) -> ChatGroq:
#     return ChatGroq(
#         groq_api_key=api_key,
#         model_name="llama-3.3-70b-versatile",
#         temperature=temperature,
#         max_tokens=4096,   # raised: full-unit summaries need more output tokens
#     )


# # ── Helper ───────────────────────────────────────────────────────────────────

# def format_context(docs: List[Document]) -> str:
#     parts = []
#     for i, doc in enumerate(docs, 1):
#         meta = doc.metadata
#         parts.append(
#             f"[Excerpt {i} | Page {meta.get('page','?')} | {meta.get('unit','?')}]\n"
#             f"{doc.page_content}"
#         )
#     return "\n\n---\n\n".join(parts)


# def _error_dict(msg: str) -> Dict:
#     return {"answer": msg, "sources": []}


# def _invoke_safe(chain, payload: dict) -> str:
#     """Invoke a LangChain chain and surface friendly errors on failure."""
#     try:
#         result = chain.invoke(payload)
#         return result.content
#     except AuthenticationError:
#         raise ValueError(
#             "❌ **Invalid Groq API key.** Please check the key you entered in the sidebar "
#             "and make sure it starts with `gsk_`."
#         )
#     except RateLimitError:
#         raise ValueError(
#             "⏳ **Groq rate limit reached.** You've hit the free-tier limit. "
#             "Wait 1–2 minutes and try again, or upgrade your Groq plan."
#         )
#     except APIError as e:
#         raise ValueError(f"🔴 **Groq API error:** {e}")


# # ── 1. General Q&A ───────────────────────────────────────────────────────────

# QA_SYSTEM = """You are a helpful and encouraging English teacher for Grade 3 students.
# Answer questions based ONLY on the provided textbook excerpts from
# "Exploring English - Grade 3" by Zahid Publications.

# STRICT RULES — follow every one without exception:

# 1. Use ONLY information explicitly present in the excerpts.
#    Do NOT infer, guess, or reason from a unit title alone.
#    Do NOT fabricate story details, character names, or plot points.

# 2. SUMMARY / OVERVIEW questions:
#    When the user asks for a summary or overview of a unit, you will be given
#    ALL the chunks of that unit in page order.  Write a complete, structured
#    summary covering: the main story/topic, key characters, important events,
#    vocabulary highlights, and any exercises or poems included.
#    Use headings like "## Story", "## Key Characters", "## Exercises" as needed.
#    Base every sentence on the text in the excerpts.

# 3. SPECIFIC questions (character names, events, fill-in-the-blank, etc.):
#    Find the exact answer in the excerpts and quote the relevant line.
#    If the answer is genuinely not in the excerpts, say:
#    "This detail isn't in the retrieved excerpts — try asking about a specific page."
#    Never guess.

# 4. The excerpts are OCR-scanned.  Headings may appear as plain capitalised
#    lines (e.g. "THE DRAWN MATCH") — treat these as unit/chapter titles.

# 5. Keep language clear, simple, and age-appropriate for Grade 3.

# 6. Cite page numbers when using specific information (e.g. "Page 6 says…").
# """

# QA_HUMAN = """Textbook excerpts (read ALL of them before answering):
# {context}

# Student's question: {question}

# Instructions:
# - If this is a SUMMARY request: write a full structured summary from ALL the excerpts above.
# - If this is a SPECIFIC question: find the exact answer in the excerpts and cite the page.
# - Do NOT guess or invent any information not present in the excerpts above."""


# def answer_question(llm: ChatGroq, context_docs: List[Document], question: str) -> Dict:
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", QA_SYSTEM),
#         ("human",  QA_HUMAN),
#     ])
#     chain = prompt | llm
#     try:
#         content = _invoke_safe(chain, {
#             "context":  format_context(context_docs),
#             "question": question,
#         })
#     except ValueError as e:
#         return _error_dict(str(e))

#     sources = list(dict.fromkeys([
#         f"Page {d.metadata.get('page','?')} ({d.metadata.get('unit','?')})"
#         for d in context_docs
#     ]))
#     return {"answer": content, "sources": sources}


# # ── 2. Exercise Q&A ──────────────────────────────────────────────────────────

# EXERCISE_SYSTEM = """You are an expert Grade 3 English teacher solving textbook exercises.
# Use the provided excerpts to answer the exercise question accurately.

# Rules:
# - Fill-in-the-blank: provide the complete sentence with blanks filled.
# - Matching: list each matched pair clearly.
# - Grammar: explain the rule briefly before answering.
# - Comprehension: answer from the passage and cite the line.
# - Always explain WHY the answer is correct so the student learns.
# """

# EXERCISE_HUMAN = """Textbook excerpts:
# {context}

# Exercise question:
# {question}

# Provide the answer with a brief explanation."""


# def answer_exercise(llm: ChatGroq, context_docs: List[Document], question: str) -> Dict:
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", EXERCISE_SYSTEM),
#         ("human",  EXERCISE_HUMAN),
#     ])
#     chain = prompt | llm
#     try:
#         content = _invoke_safe(chain, {
#             "context":  format_context(context_docs),
#             "question": question,
#         })
#     except ValueError as e:
#         return _error_dict(str(e))

#     sources = list(dict.fromkeys([
#         f"Page {d.metadata.get('page','?')} ({d.metadata.get('unit','?')})"
#         for d in context_docs
#     ]))
#     return {"answer": content, "sources": sources}


# # ── 3. Exam Paper Generation ─────────────────────────────────────────────────

# EXAM_SYSTEM = """You are an experienced Grade 3 English exam paper setter.
# Create a well-structured exam paper from the provided textbook content.

# Formatting rules:
# - Output in clean Markdown.
# - Include a header with school name placeholder, subject, grade, date, and total marks.
# - Organise questions into clearly labelled sections (Q1, Q2, ...).
# - Every question must have marks allocated (shown in brackets).
# - Questions must be directly based on the provided textbook excerpts.
# - Do NOT include an answer key in the exam paper itself.
# - End with a "Best of Luck!" footer.
# """

# EXAM_HUMAN = """Textbook content to base the exam on:
# {context}

# Exam specification:
# - Total marks: {total_marks}
# - Difficulty: {difficulty}
# - Question types to include: {q_types}
# - Focus topic/unit (if any): {focus}

# Generate the complete exam paper now."""


# def generate_exam_paper(
#     llm: ChatGroq,
#     context_docs: List[Document],
#     total_marks: int   = 50,
#     difficulty: str    = "Medium",
#     q_types: List[str] = None,
#     focus: str         = "All units",
# ) -> str:
#     if q_types is None:
#         q_types = [
#             "Multiple Choice Questions (MCQs)",
#             "Fill in the Blanks",
#             "True / False",
#             "Short Answer Questions",
#             "Write sentences using the given words",
#         ]

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", EXAM_SYSTEM),
#         ("human",  EXAM_HUMAN),
#     ])
#     chain = prompt | llm
#     try:
#         return _invoke_safe(chain, {
#             "context":     format_context(context_docs),
#             "total_marks": total_marks,
#             "difficulty":  difficulty,
#             "q_types":     ", ".join(q_types),
#             "focus":       focus,
#         })
#     except ValueError as e:
#         return str(e)


# # ── 4. Answer Key Generation ─────────────────────────────────────────────────

# ANSWER_KEY_SYSTEM = """You are an experienced Grade 3 English teacher.
# Given an exam paper and the relevant textbook content, produce a detailed answer key.
# Format it clearly in Markdown with each question's answer and a brief explanation."""

# ANSWER_KEY_HUMAN = """Textbook content:
# {context}

# Exam paper:
# {exam_paper}

# Produce a complete answer key."""


# def generate_answer_key(
#     llm: ChatGroq,
#     context_docs: List[Document],
#     exam_paper: str,
# ) -> str:
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", ANSWER_KEY_SYSTEM),
#         ("human",  ANSWER_KEY_HUMAN),
#     ])
#     chain = prompt | llm
#     try:
#         return _invoke_safe(chain, {
#             "context":    format_context(context_docs),
#             "exam_paper": exam_paper,
#         })
#     except ValueError as e:
#         return str(e)





"""
LLM Chains powered by Groq (llama-3.3-70b-versatile).

FIXED VERSION — key changes vs original:
  1. Every prompt now includes the FULL unit directory (unit names + page ranges)
     so the LLM always knows all 10 units and never has to guess.
  2. QA system prompt strengthened: LLM is explicitly told NOT to say
     "I don't have the full text" when it has the relevant excerpts.
  3. answer_question now passes the unit directory into the system prompt.
  4. Graceful error handling retained from original.
"""

from typing import List, Dict

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

try:
    from groq import RateLimitError, AuthenticationError, APIError
except ImportError:
    from groq._exceptions import RateLimitError, AuthenticationError, APIError

# Import the unit directory from rag_engine so prompts stay in sync
try:
    from rag_engine import get_all_units_summary, UNIT_PAGE_MAP
except ImportError:
    # Fallback if rag_engine not importable yet
    get_all_units_summary = lambda: "(unit list unavailable)"
    UNIT_PAGE_MAP = []


# ── LLM factory ───────────────────────────────────────────────────────────────

def get_llm(api_key: str, temperature: float = 0.3) -> ChatGroq:
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=2048,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── 1. General Q&A ────────────────────────────────────────────────────────────

QA_SYSTEM = """You are a knowledgeable and encouraging English teacher for Grade 3 students.
You are answering questions about the textbook "Exploring English - Grade 3" by Zahid Publications.

TEXTBOOK STRUCTURE — memorise this, it is ground truth:
{unit_directory}

RULES:
1. Answer ONLY from the provided excerpts. Do not invent content.
2. The excerpts include a header like [Unit X: Title | pages N–M | Page P].
   Use this to identify which unit each excerpt belongs to.
3. When asked for a unit summary or content, synthesise from ALL excerpts that 
   belong to that unit — do not say "I only have pages X and Y". 
   Instead say "Based on the excerpts from this unit..."
4. Never say "I don't have the full text" or "Unfortunately we don't have excerpts" 
   if ANY excerpt from that unit is present. Work with what you have.
5. Keep answers clear, simple, and age-appropriate (Grade 3 level).
6. Always mention the unit title and relevant page numbers in your answer.
7. Quote relevant parts of the text when helpful.
"""

QA_HUMAN = """Textbook excerpts:
{context}

Student's question: {question}

Answer clearly and helpfully using the excerpts above."""


def answer_question(llm: ChatGroq, context_docs: List[Document], question: str) -> Dict:
    unit_dir = get_all_units_summary()
    prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM),
        ("human",  QA_HUMAN),
    ])
    chain = prompt | llm
    try:
        content = _invoke_safe(chain, {
            "unit_directory": unit_dir,
            "context":        format_context(context_docs),
            "question":       question,
        })
    except ValueError as e:
        return _error_dict(str(e))

    sources = list(dict.fromkeys([
        f"Page {d.metadata.get('page','?')} ({d.metadata.get('unit','?')})"
        for d in context_docs
    ]))
    return {"answer": content, "sources": sources}


# ── 2. Exercise Q&A ───────────────────────────────────────────────────────────

EXERCISE_SYSTEM = """You are an expert Grade 3 English teacher solving textbook exercises.
Use the provided excerpts from "Exploring English - Grade 3" (Zahid Publications) to answer
the exercise question accurately.

TEXTBOOK STRUCTURE:
{unit_directory}

RULES:
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
    unit_dir = get_all_units_summary()
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXERCISE_SYSTEM),
        ("human",  EXERCISE_HUMAN),
    ])
    chain = prompt | llm
    try:
        content = _invoke_safe(chain, {
            "unit_directory": unit_dir,
            "context":        format_context(context_docs),
            "question":       question,
        })
    except ValueError as e:
        return _error_dict(str(e))

    sources = list(dict.fromkeys([
        f"Page {d.metadata.get('page','?')} ({d.metadata.get('unit','?')})"
        for d in context_docs
    ]))
    return {"answer": content, "sources": sources}


# ── 3. Exam Paper Generation ──────────────────────────────────────────────────

EXAM_SYSTEM = """You are an experienced Grade 3 English exam paper setter.
Create a well-structured exam paper from the provided textbook content.

TEXTBOOK STRUCTURE:
{unit_directory}

FORMATTING RULES:
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
    total_marks: int = 50,
    difficulty: str = "Medium",
    q_types: List[str] = None,
    focus: str = "All units",
) -> str:
    if q_types is None:
        q_types = [
            "Multiple Choice Questions (MCQs)",
            "Fill in the Blanks",
            "True / False",
            "Short Answer Questions",
            "Write sentences using the given words",
        ]

    unit_dir = get_all_units_summary()
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXAM_SYSTEM),
        ("human",  EXAM_HUMAN),
    ])
    chain = prompt | llm
    try:
        return _invoke_safe(chain, {
            "unit_directory": unit_dir,
            "context":        format_context(context_docs),
            "total_marks":    total_marks,
            "difficulty":     difficulty,
            "q_types":        ", ".join(q_types),
            "focus":          focus,
        })
    except ValueError as e:
        return str(e)


# ── 4. Answer Key Generation ──────────────────────────────────────────────────

ANSWER_KEY_SYSTEM = """You are an experienced Grade 3 English teacher.
Given an exam paper and the relevant textbook content, produce a detailed answer key.
Format it clearly in Markdown with each question's answer and a brief explanation.

TEXTBOOK STRUCTURE:
{unit_directory}
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
    unit_dir = get_all_units_summary()
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_KEY_SYSTEM),
        ("human",  ANSWER_KEY_HUMAN),
    ])
    chain = prompt | llm
    try:
        return _invoke_safe(chain, {
            "unit_directory": unit_dir,
            "context":        format_context(context_docs),
            "exam_paper":     exam_paper,
        })
    except ValueError as e:
        return str(e)
