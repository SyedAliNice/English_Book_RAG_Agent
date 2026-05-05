"""
diagnose.py  —  Run this ONCE to find out what's broken.

Usage:
    python diagnose.py

It will check:
  1. Is Tesseract installed and accessible?
  2. How much text does OCR actually extract from the first 3 pages?
  3. How many documents are in the ChromaDB index?
  4. What do retrieved chunks actually look like for a sample query?
"""

import sys
from pathlib import Path

PDF_PATH    = "english_3.pdf"
PERSIST_DIR = "chroma_db"

SEP = "─" * 60

# ── 1. Tesseract check ────────────────────────────────────────────────────────
print(SEP)
print("STEP 1: Tesseract")
try:
    import pytesseract
    ver = pytesseract.get_tesseract_version()
    print(f"  ✅ Tesseract found: version {ver}")
except Exception as e:
    print(f"  ❌ Tesseract NOT found: {e}")
    print("     → Install from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("     → Then add its folder to PATH, or set pytesseract.pytesseract.tesseract_cmd")
    sys.exit(1)

# ── 2. OCR sample — first 3 pages ────────────────────────────────────────────
print(SEP)
print("STEP 2: OCR extraction from first 3 pages of PDF")

if not Path(PDF_PATH).exists():
    print(f"  ❌ PDF not found at '{PDF_PATH}'. Place it next to this script.")
    sys.exit(1)

import io
import fitz
from PIL import Image

doc = fitz.open(PDF_PATH)
print(f"  PDF has {len(doc)} pages.")

for page_num in range(min(3, len(doc))):
    page = doc[page_num]

    # Native text
    native = page.get_text("text").strip()

    # OCR
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    ocr_text = pytesseract.image_to_string(img, lang="eng").strip()

    print(f"\n  ── Page {page_num+1} ──")
    print(f"  Native text length : {len(native)} chars")
    print(f"  OCR text length    : {len(ocr_text)} chars")
    if ocr_text:
        print(f"  OCR preview        : {repr(ocr_text[:200])}")
    else:
        print("  ⚠️  OCR returned EMPTY string for this page!")

doc.close()

# ── 3. ChromaDB contents ──────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 3: ChromaDB index contents")

if not Path(PERSIST_DIR).exists():
    print(f"  ❌ No '{PERSIST_DIR}' folder found — index was never built.")
else:
    try:
        from sentence_transformers import SentenceTransformer
        from langchain_core.embeddings import Embeddings
        from typing import List

        class _Emb(Embeddings):
            def __init__(self):
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            def embed_documents(self, texts):
                return self.model.encode(texts, normalize_embeddings=True).tolist()
            def embed_query(self, text):
                return self.model.encode(text, normalize_embeddings=True).tolist()

        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma

        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=_Emb())
        col = db._collection
        count = col.count()
        print(f"  Documents in index: {count}")

        if count == 0:
            print("  ❌ Index is EMPTY — rebuild it with the 🔄 button.")
        else:
            # Peek at a few raw docs
            sample = col.peek(3)
            for i, (doc_id, text, meta) in enumerate(
                zip(sample["ids"], sample["documents"], sample["metadatas"])
            ):
                print(f"\n  ── Chunk {i+1} (id={doc_id}) ──")
                print(f"  Metadata : {meta}")
                print(f"  Text len : {len(text)} chars")
                print(f"  Preview  : {repr(text[:300])}")

    except Exception as e:
        print(f"  ❌ Could not read ChromaDB: {e}")

# ── 4. Retrieval test ─────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 4: Retrieval test — query: 'unit 1 name'")

try:
    results = db.similarity_search("unit 1 name", k=3)
    if not results:
        print("  ❌ No results returned for query!")
    for i, r in enumerate(results):
        print(f"\n  Result {i+1}: page={r.metadata.get('page','?')} unit={r.metadata.get('unit','?')}")
        print(f"  Text: {repr(r.page_content[:300])}")
except Exception as e:
    print(f"  ❌ Retrieval failed: {e}")

print()
print(SEP)
print("DIAGNOSIS COMPLETE — share the output above to identify the root cause.")
print(SEP)
