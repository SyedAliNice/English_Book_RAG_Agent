"""
RAG Engine for Exploring English Grade 3
=========================================
Uses EasyOCR instead of Tesseract — pure pip install, no system setup needed.

    pip install easyocr

EasyOCR downloads its model (~100 MB) on first use automatically.
"""

import io
import re
from pathlib import Path
from typing import List

import fitz                 # PyMuPDF  (pip install pymupdf)
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


# ── Constants ────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
PERSIST_DIR   = "chroma_db"
RENDER_DPI    = 2.0   # 2.0 ≈ 150 DPI — good balance of speed vs quality

# Unit-heading patterns (digit form, word form, chapter, numbered title)
_UNIT_PATTERNS = [
    re.compile(r"Unit\s+\d+",                                          re.IGNORECASE),
    re.compile(r"Unit\s+(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)",
               re.IGNORECASE),
    re.compile(r"Chapter\s+\d+",                                       re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*\d+\.\s+[A-Z][A-Za-z ]{4,40}"),
]


# ── EasyOCR singleton (loaded once, reused for all pages) ────────────────────
_easyocr_reader = None

def _get_ocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        # gpu=False works on any machine; change to True if you have CUDA
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


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


# ── Page rendering ────────────────────────────────────────────────────────────

def _render_page(page: fitz.Page, scale: float = RENDER_DPI) -> np.ndarray:
    """Render a PDF page to a numpy RGB array (what EasyOCR expects)."""
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return np.array(img)


# ── OCR ───────────────────────────────────────────────────────────────────────

def _ocr_page(img_array: np.ndarray) -> str:
    """Run EasyOCR on a numpy image array and return joined text."""
    reader  = _get_ocr_reader()
    results = reader.readtext(img_array, detail=0, paragraph=True)
    return "\n".join(results)


# ── Unit detection ────────────────────────────────────────────────────────────

def _detect_unit(text: str, window: int = 500) -> str | None:
    snippet = text[:window]
    for pat in _UNIT_PATTERNS:
        m = pat.search(snippet)
        if m:
            return m.group(1) if pat.groups else m.group(0)
    return None


# ── PDF Extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str, status_callback=None) -> List[Document]:
    """
    Extract text from every page.
    Tries native PDF text first (fast); falls back to EasyOCR for image-based pages.
    status_callback(page_num, total) — optional Streamlit progress hook.
    """
    doc           = fitz.open(pdf_path)
    total         = len(doc)
    documents     = []
    current_unit  = "Introduction"

    for page_num in range(total):
        if status_callback:
            status_callback(page_num + 1, total)

        page        = doc[page_num]
        native_text = page.get_text("text").strip()

        if len(native_text) > 50:
            text = native_text
        else:
            img_array = _render_page(page)
            text      = _ocr_page(img_array).strip()

        if not text:
            continue

        detected = _detect_unit(text)
        if detected:
            current_unit = detected

        # Prepend unit label so the LLM sees it inside every retrieved chunk
        labeled_text = f"[{current_unit}]\n{text}"

        documents.append(Document(
            page_content=labeled_text,
            metadata={
                "page":   page_num + 1,
                "unit":   current_unit,
                "source": Path(pdf_path).name,
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
    return [c for c in chunks if c.page_content.strip()]


# ── Vector Store ──────────────────────────────────────────────────────────────

def build_vector_store(chunks: List[Document], persist_dir: str = PERSIST_DIR) -> Chroma:
    embeddings = get_embeddings()
    BATCH      = 100
    vectordb   = None

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
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


# ── Top-level initialiser ─────────────────────────────────────────────────────

def initialise_rag(pdf_path: str, force_rebuild: bool = False,
                   status_callback=None) -> Chroma:
    if not force_rebuild and vector_store_exists():
        return load_vector_store()

    raw_docs = extract_text_from_pdf(pdf_path, status_callback=status_callback)
    chunks   = split_documents(raw_docs)

    if not chunks:
        raise ValueError(
            "No text could be extracted from the PDF. "
            "Make sure easyocr is installed: pip install easyocr"
        )

    return build_vector_store(chunks)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_context(vectordb: Chroma, query: str, k: int = 6) -> List[Document]:
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20},
    )
    return retriever.invoke(query)