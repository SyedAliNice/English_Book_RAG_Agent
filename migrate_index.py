"""
migrate_index.py — Run this ONCE after replacing rag_engine.py and llm_chains.py.

It deletes the old ChromaDB index (which was built with wrong unit labels)
and rebuilds it with the corrected unit metadata.

Usage:
    python migrate_index.py english_3.pdf
"""

import sys
import shutil
from pathlib import Path

PDF_PATH    = sys.argv[1] if len(sys.argv) > 1 else "english_3.pdf"
PERSIST_DIR = "chroma_db"

print("=" * 60)
print("MIGRATION: Rebuilding ChromaDB with fixed unit metadata")
print("=" * 60)

# 1. Verify PDF exists
if not Path(PDF_PATH).exists():
    print(f"❌ PDF not found at '{PDF_PATH}'. Pass the path as argument:")
    print(f"   python migrate_index.py path/to/english_3.pdf")
    sys.exit(1)

print(f"✅ PDF found: {PDF_PATH}")

# 2. Print the unit map so you can verify it
from rag_engine import UNIT_PAGE_MAP, get_all_units_summary
print("\nUnit map that will be used:")
print(get_all_units_summary())

# 3. Delete old index
if Path(PERSIST_DIR).exists():
    shutil.rmtree(PERSIST_DIR)
    print(f"\n🗑️  Deleted old index at '{PERSIST_DIR}'")
else:
    print(f"\nℹ️  No existing index found at '{PERSIST_DIR}' — building fresh.")

# 4. Build new index
print("\n⏳ Building new index (this may take 2–5 minutes for 142 pages)...\n")

from rag_engine import extract_text_from_pdf, split_documents, build_vector_store

def progress(page, total):
    pct = int(page / total * 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct}% — page {page}/{total}", end="", flush=True)

raw_docs = extract_text_from_pdf(PDF_PATH, status_callback=progress)
print(f"\n✅ Extracted {len(raw_docs)} pages")

chunks = split_documents(raw_docs)
print(f"✅ Created {len(chunks)} chunks")

# Spot-check first 3 chunks
print("\nSample chunks (first 3):")
for i, c in enumerate(chunks[:3]):
    print(f"  Chunk {i+1}: unit={c.metadata.get('unit','?')} | page={c.metadata.get('page','?')}")
    print(f"    Preview: {repr(c.page_content[:120])}")
    print()

vectordb = build_vector_store(chunks)
print(f"✅ Index built and saved to '{PERSIST_DIR}'")

# 5. Quick retrieval test
print("\nRetrieval test — query: 'unit 1 The Drawn Match Shahid'")
from rag_engine import retrieve_context
results = retrieve_context(vectordb, "unit 1 The Drawn Match Shahid", k=4)
for i, r in enumerate(results):
    print(f"  Result {i+1}: {r.metadata.get('unit','?')} | page {r.metadata.get('page','?')}")
    print(f"    {repr(r.page_content[:120])}")

print("\n" + "=" * 60)
print("MIGRATION COMPLETE — restart your Streamlit app now.")
print("=" * 60)
