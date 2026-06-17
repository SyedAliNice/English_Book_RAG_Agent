# """
# RAG Engine for Exploring English Grade 3
# =========================================
# Uses EasyOCR instead of Tesseract — pure pip install, no system setup needed.

#     pip install easyocr

# EasyOCR downloads its model (~100 MB) on first use automatically.
# """

# import io
# import re
# from pathlib import Path
# from typing import List

# import fitz                 # PyMuPDF  (pip install pymupdf)
# import numpy as np
# from PIL import Image

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from langchain_core.embeddings import Embeddings
# from sentence_transformers import SentenceTransformer

# try:
#     from langchain_chroma import Chroma
# except ImportError:
#     from langchain_community.vectorstores import Chroma  # type: ignore


# # ── Constants ────────────────────────────────────────────────────────────────
# EMBED_MODEL   = "all-MiniLM-L6-v2"
# # FIX 3 — larger chunks so each retrieved excerpt contains more of a unit's
# # content, reducing the chance that the LLM has to guess across gaps.
# # Overlap kept proportionally high (30 %) so unit headings that appear near
# # a chunk boundary are included in both neighbouring chunks.
# CHUNK_SIZE    = 1500
# CHUNK_OVERLAP = 400
# PERSIST_DIR   = "chroma_db"
# RENDER_DPI    = 2.0   # 2.0 ≈ 150 DPI — good balance of speed vs quality

# # Unit-heading patterns (digit form, word form, chapter, numbered title)
# #
# # FIX 2 — expanded patterns:
# #   OCR on a Grade 3 textbook often produces heading variants such as:
# #     "Unit 1"  /  "UNIT ONE"  /  "Unit: 1"  /  "Unit-1"
# #   The original patterns missed these.  The new set is deliberately
# #   broad so that any reasonable OCR rendering of a unit heading is caught.
# #   Patterns are tried in order; the first match wins.
# _UNIT_PATTERNS = [
#     # "Unit 1", "UNIT 2", "unit 10"  (digit, optional separator)
#     re.compile(r"\bUnit[\s:.\-]*(\d{1,2})\b",                          re.IGNORECASE),
#     # "Unit One" … "Unit Ten"
#     re.compile(r"\bUnit[\s:.\-]*"
#                r"(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\b",
#                re.IGNORECASE),
#     # "Chapter 3", "Chapter Three"
#     re.compile(r"\bChapter[\s:.\-]*"
#                r"(\d{1,2}|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\b",
#                re.IGNORECASE),
#     # Numbered section titles like "1. The Drawn Match"
#     re.compile(r"(?:^|\n)\s*(\d{1,2})\.\s+[A-Z][A-Za-z ]{4,40}"),
#     # ALL-CAPS heading on its own line, at least 4 chars (e.g. "THE DRAWN MATCH")
#     re.compile(r"(?:^|\n)([A-Z][A-Z ]{3,40})(?:\n|$)"),
# ]


# # ── EasyOCR singleton (loaded once, reused for all pages) ────────────────────
# _easyocr_reader = None

# def _get_ocr_reader():
#     global _easyocr_reader
#     if _easyocr_reader is None:
#         import easyocr
#         # gpu=False works on any machine; change to True if you have CUDA
#         _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
#     return _easyocr_reader


# # ── Embedding Wrapper ─────────────────────────────────────────────────────────

# class DirectSTEmbeddings(Embeddings):
#     def __init__(self, model_name: str = EMBED_MODEL):
#         self.model = SentenceTransformer(model_name)

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         if not texts:
#             return []
#         return self.model.encode(
#             texts, normalize_embeddings=True, show_progress_bar=False
#         ).tolist()

#     def embed_query(self, text: str) -> List[float]:
#         return self.model.encode(
#             text, normalize_embeddings=True, show_progress_bar=False
#         ).tolist()


# def get_embeddings() -> DirectSTEmbeddings:
#     return DirectSTEmbeddings(EMBED_MODEL)


# # ── Page rendering ────────────────────────────────────────────────────────────

# def _render_page(page: fitz.Page, scale: float = RENDER_DPI) -> np.ndarray:
#     """Render a PDF page to a numpy RGB array (what EasyOCR expects)."""
#     mat = fitz.Matrix(scale, scale)
#     pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
#     img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
#     return np.array(img)


# # ── OCR ───────────────────────────────────────────────────────────────────────

# def _ocr_page(img_array: np.ndarray) -> str:
#     """Run EasyOCR on a numpy image array and return joined text."""
#     reader  = _get_ocr_reader()
#     results = reader.readtext(img_array, detail=0, paragraph=True)
#     return "\n".join(results)


# # ── Unit detection ────────────────────────────────────────────────────────────

# def _detect_unit(text: str, window: int = 600) -> str | None:
#     """
#     Return the unit/chapter label found in the first `window` chars of text,
#     or None if no heading is detected.

#     FIX 2 — uses the full match (group 0) so that labels like "Unit 1" or
#     "Chapter Two" are stored in metadata, not just the capture group digit.
#     The wider search window (600 vs 500) catches headings that appear after
#     a short OCR-generated page-number line.
#     """
#     snippet = text[:window]
#     for pat in _UNIT_PATTERNS:
#         m = pat.search(snippet)
#         if m:
#             return m.group(0).strip()
#     return None


# # ── PDF Extraction ────────────────────────────────────────────────────────────

# def extract_text_from_pdf(pdf_path: str, status_callback=None) -> List[Document]:
#     """
#     Extract text from every page.
#     Tries native PDF text first (fast); falls back to EasyOCR for image-based pages.
#     status_callback(page_num, total) — optional Streamlit progress hook.
#     """
#     doc           = fitz.open(pdf_path)
#     total         = len(doc)
#     documents     = []
#     current_unit  = "Introduction"

#     for page_num in range(total):
#         if status_callback:
#             status_callback(page_num + 1, total)

#         page        = doc[page_num]
#         native_text = page.get_text("text").strip()

#         if len(native_text) > 50:
#             text = native_text
#         else:
#             img_array = _render_page(page)
#             text      = _ocr_page(img_array).strip()

#         if not text:
#             continue

#         detected = _detect_unit(text)
#         if detected:
#             current_unit = detected

#         # Prepend unit label so the LLM sees it inside every retrieved chunk
#         labeled_text = f"[{current_unit}]\n{text}"

#         documents.append(Document(
#             page_content=labeled_text,
#             metadata={
#                 "page":   page_num + 1,
#                 "unit":   current_unit,
#                 "source": Path(pdf_path).name,
#             },
#         ))

#     doc.close()
#     return documents


# # ── Chunking ──────────────────────────────────────────────────────────────────

# def split_documents(documents: List[Document]) -> List[Document]:
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ". ", " ", ""],
#     )
#     chunks = splitter.split_documents(documents)
#     return [c for c in chunks if c.page_content.strip()]


# # ── Vector Store ──────────────────────────────────────────────────────────────

# def build_vector_store(chunks: List[Document], persist_dir: str = PERSIST_DIR) -> Chroma:
#     embeddings = get_embeddings()
#     BATCH      = 100
#     vectordb   = None

#     for i in range(0, len(chunks), BATCH):
#         batch = chunks[i : i + BATCH]
#         if vectordb is None:
#             vectordb = Chroma.from_documents(
#                 documents=batch,
#                 embedding=embeddings,
#                 persist_directory=persist_dir,
#             )
#         else:
#             vectordb.add_documents(batch)

#     vectordb.persist()
#     return vectordb


# def load_vector_store(persist_dir: str = PERSIST_DIR) -> Chroma:
#     return Chroma(
#         persist_directory=persist_dir,
#         embedding_function=get_embeddings(),
#     )


# def vector_store_exists(persist_dir: str = PERSIST_DIR) -> bool:
#     p = Path(persist_dir)
#     return p.exists() and any(p.iterdir())


# # ── Top-level initialiser ─────────────────────────────────────────────────────

# def initialise_rag(pdf_path: str, force_rebuild: bool = False,
#                    status_callback=None) -> Chroma:
#     if not force_rebuild and vector_store_exists():
#         return load_vector_store()

#     raw_docs = extract_text_from_pdf(pdf_path, status_callback=status_callback)
#     chunks   = split_documents(raw_docs)

#     if not chunks:
#         raise ValueError(
#             "No text could be extracted from the PDF. "
#             "Make sure easyocr is installed: pip install easyocr"
#         )

#     return build_vector_store(chunks)


# # ── Unit number / name normalisation ─────────────────────────────────────────

# _WORD_TO_NUM = {
#     "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
#     "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
# }

# # Matches "unit 1", "unit one", "unit no 1", "unit number 2", "unit no. 3"
# _QUERY_UNIT_RE = re.compile(
#     r"\bunit\s*(?:no\.?\s*|number\s*)?(\d{1,2}|one|two|three|four|five|"
#     r"six|seven|eight|nine|ten)\b",
#     re.IGNORECASE,
# )


# def _extract_unit_number(query: str):
#     """Return canonical unit digit string ('1', '2', …) from a query, or None."""
#     m = _QUERY_UNIT_RE.search(query)
#     if not m:
#         return None
#     raw = m.group(1).lower()
#     return _WORD_TO_NUM.get(raw, raw)


# def _is_unit_wide_query(query: str) -> bool:
#     """
#     Return True when the question is about an entire unit rather than a
#     specific detail within it.
#     """
#     broad_keywords = re.compile(
#         r"\b(summar(?:y|ise|ize)|overview|about|describe|explain|"
#         r"tell\s+me|read\s+(?:the\s+)?(?:entire\s+)?unit|what\s+is)\b",
#         re.IGNORECASE,
#     )
#     return bool(broad_keywords.search(query))


# # ── Metadata-based full-unit fetch ────────────────────────────────────────────

# def _fetch_unit_chunks(vectordb: Chroma, unit_number: str) -> List[Document]:
#     """
#     Pull every stored chunk whose 'unit' metadata contains the target unit
#     number, then sort them by page so the LLM receives them in reading order.
#     This bypasses embedding similarity entirely and guarantees complete
#     coverage of the unit regardless of how the question is phrased.
#     """
#     collection = vectordb._collection
#     results    = collection.get(include=["documents", "metadatas"])

#     matched: List[Document] = []
#     unit_variants = re.compile(
#         rf"\b(?:unit\s*(?:no\.?\s*)?)?{re.escape(unit_number)}\b",
#         re.IGNORECASE,
#     )
#     for text, meta in zip(results["documents"], results["metadatas"]):
#         stored_unit = str(meta.get("unit", ""))
#         if unit_variants.search(stored_unit):
#             matched.append(Document(page_content=text, metadata=meta))

#     # Sort by page number so the LLM reads the unit in order
#     matched.sort(key=lambda d: int(d.metadata.get("page", 0)))
#     return matched


# # ── Retrieval ─────────────────────────────────────────────────────────────────

# def retrieve_context(vectordb: Chroma, query: str, k: int = 10) -> List[Document]:
#     """
#     Smart retrieval: unit-wide questions fetch ALL chunks for that unit by
#     metadata filter (sorted by page); specific-detail questions use similarity.

#     Why: similarity search for "summary of unit 1" pulls 10 chunks from
#     across the whole book.  The LLM gets an incomplete picture and either
#     hallucinates or says "not in excerpts".  Fetching all unit chunks by
#     metadata gives it the complete, ordered text it needs.
#     """
#     unit_num = _extract_unit_number(query)

#     if unit_num and _is_unit_wide_query(query):
#         chunks = _fetch_unit_chunks(vectordb, unit_num)
#         if chunks:
#             return chunks
#         # Fall through to similarity if metadata fetch returns nothing
#         # (index built before unit-tagging fix)

#     retriever = vectordb.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": k},
#     )
#     return retriever.invoke(query)




"""
RAG Engine for Exploring English Grade 3
=========================================
FIXED VERSION — addresses inconsistent unit detection & retrieval.

Key changes vs original:
  1. HARDCODED unit page map — no more OCR-based guessing of unit names.
     The PDF's structure is fixed, so we hardcode the ground truth.
  2. Switched OCR backend to pytesseract (faster, available on most systems).
     EasyOCR is retained as a fallback.
  3. Larger default retrieval k=8 (was 6).
  4. Chunk size increased to 1000 (was 800) — captures more context per chunk.
  5. Unit label prepended as a structured header so LLM always knows the unit.
  6. Page-range metadata added so LLM can say "Unit 3 spans pages 29–43".
"""

import io
import re
from pathlib import Path
from typing import List, Optional

import fitz                       # PyMuPDF
import numpy as np
from PIL import Image

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # type: ignore


# ── Constants ─────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 1000   # ↑ more context per chunk
CHUNK_OVERLAP = 200
PERSIST_DIR   = "chroma_db"
RENDER_DPI    = 2.0    # ≈150 DPI for OCR

# ─────────────────────────────────────────────────────────────────────────────
# HARDCODED UNIT MAP  (page numbers are 1-based, from the actual PDF scan)
# Format: (start_page, end_page, unit_number, unit_title)
# This eliminates all OCR-based unit-detection errors.
# ─────────────────────────────────────────────────────────────────────────────
UNIT_PAGE_MAP = [
    (1,  3,   0,  "Front Matter / Preface"),
    (4,  16,  1,  "Unit 1: The Drawn Match"),
    (17, 28,  2,  "Unit 2: The Joy of Helping Others"),
    (29, 43,  3,  "Unit 3: My Village"),
    (44, 55,  4,  "Unit 4: Animal Friends (Poem)"),
    (56, 70,  5,  "Unit 5: All are Equal"),
    (71, 86,  6,  "Unit 6: Hazrat Umar (RA)"),
    (87, 100, 7,  "Unit 7: The Uses of Mobile Phones"),
    (101, 115, 8, "Unit 8: Common Professions in Pakistan"),
    (116, 128, 9, "Unit 9: Keep Our World Clean (Poem)"),
    (129, 142, 10, "Unit 10: Staying Safe at Home"),
]


def _page_to_unit(page_num: int) -> tuple[str, int]:
    """Return (unit_label, unit_number) for a 1-based page number."""
    for start, end, num, label in UNIT_PAGE_MAP:
        if start <= page_num <= end:
            return label, num
    return "Unknown Section", -1


# ── OCR — pytesseract primary, EasyOCR fallback ───────────────────────────────

def _ocr_with_tesseract(img: Image.Image) -> str:
    import pytesseract
    return pytesseract.image_to_string(img, lang="eng", config="--psm 6").strip()


_easyocr_reader = None

def _ocr_with_easyocr(img_array: np.ndarray) -> str:
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = _easyocr_reader.readtext(img_array, detail=0, paragraph=True)
    return "\n".join(results)


def _ocr_page(page: fitz.Page) -> str:
    """Render a PDF page and extract text via OCR."""
    mat = fitz.Matrix(RENDER_DPI, RENDER_DPI)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_pil = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    # Try pytesseract first (much faster)
    try:
        text = _ocr_with_tesseract(img_pil)
        if text:
            return text
    except Exception:
        pass

    # Fall back to EasyOCR
    try:
        img_np = np.array(img_pil)
        return _ocr_with_easyocr(img_np)
    except Exception:
        return ""


# ── Embedding Wrapper ─────────────────────────────────────────────────────────

class DirectSTEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        ).tolist()


def get_embeddings() -> DirectSTEmbeddings:
    return DirectSTEmbeddings(EMBED_MODEL)


# ── PDF Extraction ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str, status_callback=None) -> List[Document]:
    """
    Extract text from every page using OCR.
    Unit labels come from UNIT_PAGE_MAP (hardcoded ground truth),
    NOT from OCR text — this is the key fix for wrong unit detection.
    """
    doc = fitz.open(pdf_path)
    total = len(doc)
    documents = []

    for page_num in range(total):
        if status_callback:
            status_callback(page_num + 1, total)

        page = doc[page_num]
        page_1based = page_num + 1

        # Try native text first (fast)
        native_text = page.get_text("text").strip()
        text = native_text if len(native_text) > 50 else _ocr_page(page).strip()

        if not text:
            continue

        # Get unit from hardcoded map — never from OCR guessing
        unit_label, unit_num = _page_to_unit(page_1based)

        # Rich header so every chunk carries full context
        # even when the chunk is retrieved in isolation
        unit_range_str = ""
        for start, end, num, label in UNIT_PAGE_MAP:
            if num == unit_num:
                unit_range_str = f"pages {start}–{end}"
                break

        labeled_text = (
            f"[{unit_label} | {unit_range_str} | Page {page_1based}]\n"
            f"{text}"
        )

        documents.append(Document(
            page_content=labeled_text,
            metadata={
                "page":       page_1based,
                "unit":       unit_label,
                "unit_num":   unit_num,
                "unit_range": unit_range_str,
                "source":     Path(pdf_path).name,
            },
        ))

    doc.close()
    return documents


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    # Re-inject unit header into every chunk so it's never lost after splitting
    enriched = []
    for chunk in chunks:
        if not chunk.page_content.strip():
            continue
        meta = chunk.metadata
        if not chunk.page_content.startswith("["):
            header = (
                f"[{meta.get('unit','?')} | {meta.get('unit_range','?')} "
                f"| Page {meta.get('page','?')}]\n"
            )
            chunk.page_content = header + chunk.page_content
        enriched.append(chunk)
    return enriched


# ── Vector Store ──────────────────────────────────────────────────────────────

def build_vector_store(chunks: List[Document], persist_dir: str = PERSIST_DIR) -> Chroma:
    embeddings = get_embeddings()
    BATCH = 100
    vectordb = None

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i: i + BATCH]
        if vectordb is None:
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_dir,
            )
        else:
            vectordb.add_documents(batch)

    vectordb.persist()
    return vectordb


def load_vector_store(persist_dir: str = PERSIST_DIR) -> Chroma:
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),
    )


def vector_store_exists(persist_dir: str = PERSIST_DIR) -> bool:
    p = Path(persist_dir)
    return p.exists() and any(p.iterdir())


# ── Top-level initialiser ──────────────────────────────────────────────────────

def initialise_rag(pdf_path: str, force_rebuild: bool = False,
                   status_callback=None) -> Chroma:
    if not force_rebuild and vector_store_exists():
        return load_vector_store()

    raw_docs = extract_text_from_pdf(pdf_path, status_callback=status_callback)
    chunks = split_documents(raw_docs)

    if not chunks:
        raise ValueError(
            "No text could be extracted from the PDF. "
            "Ensure pytesseract (and Tesseract) or easyocr is installed."
        )

    return build_vector_store(chunks)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_context(vectordb: Chroma, query: str, k: int = 8) -> List[Document]:
    """
    Retrieve relevant chunks with two strategies:

    Strategy A — Unit-specific query (e.g. "summarize unit 3", "what is in unit 7"):
        Use Chroma's WHERE metadata filter on unit_num so we pull chunks
        DIRECTLY from that unit's pages, bypassing semantic similarity entirely.
        Semantic search cannot reliably find units with generic titles like
        "My Village", "Animal Friends", "Mobile Phones", or "Keep Our World Clean"
        because those words appear scattered across the whole book.

    Strategy B — General query (no unit number mentioned):
        Standard MMR semantic search across the full index.
    """
    # Detect if query targets a specific unit number
    unit_match = re.search(r'\bunit\s+(?:no\.?\s*)?(\d+)\b', query, re.IGNORECASE)

    if unit_match:
        target_unit_num = int(unit_match.group(1))
        target_label = None
        for start, end, num, label in UNIT_PAGE_MAP:
            if num == target_unit_num:
                target_label = label
                break

        if target_label is not None:
            # ── Strategy A: metadata-filtered retrieval ────────────────────
            # Pull all chunks belonging to this unit, then score by similarity
            # to the query and return the top-k. This guarantees we never miss
            # a unit just because its title uses common English words.
            try:
                # Fetch all chunks for this unit via metadata filter
                all_unit_chunks = vectordb.get(
                    where={"unit_num": target_unit_num},
                    include=["documents", "metadatas", "embeddings"],
                )

                if all_unit_chunks and all_unit_chunks.get("documents"):
                    docs = []
                    for doc_text, meta in zip(
                        all_unit_chunks["documents"],
                        all_unit_chunks["metadatas"],
                    ):
                        docs.append(Document(page_content=doc_text, metadata=meta))

                    # If we have more than k chunks, score by similarity and pick top-k
                    if len(docs) > k:
                        query_embedding = vectordb._embedding_function.embed_query(query)
                        import numpy as np
                        scored = []
                        embeddings = all_unit_chunks.get("embeddings") or []
                        if embeddings:
                            for doc, emb in zip(docs, embeddings):
                                score = float(np.dot(query_embedding, emb))
                                scored.append((score, doc))
                            scored.sort(key=lambda x: x[0], reverse=True)
                            docs = [d for _, d in scored[:k]]
                        else:
                            docs = docs[:k]

                    return docs

            except Exception:
                # Fall through to semantic search if metadata filter fails
                pass

    # ── Strategy B: standard MMR semantic search ───────────────────────────
    # Enrich the query with the unit title/page range if a unit was mentioned
    enriched_query = query
    if unit_match:
        unit_num = int(unit_match.group(1))
        for start, end, num, label in UNIT_PAGE_MAP:
            if num == unit_num:
                enriched_query = f"{query} {label} pages {start} to {end}"
                break

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 40},
    )
    return retriever.invoke(enriched_query)


# ── Utility: unit info for prompts ────────────────────────────────────────────

def get_all_units_summary() -> str:
    """Return a compact unit directory string to inject into LLM prompts."""
    lines = []
    for start, end, num, label in UNIT_PAGE_MAP:
        if num > 0:
            lines.append(f"  • {label} (pages {start}–{end})")
    return "\n".join(lines)
