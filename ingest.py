import os
import re
import time
from pathlib import Path

import fitz  # pymupdf
from google import genai
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY")
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
EMBED_MODEL   = "text-embedding-004"

def extract_text(pdf_path):
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            pages.append(text)
    doc.close()
    return "\n".join(pages)

def make_chunks(text):
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        last_period = chunk.rfind(". ")
        if last_period > CHUNK_SIZE * 0.6:
            chunk = chunk[:last_period + 1]
            end = start + last_period + 1
        chunk = chunk.strip()
        if chunk:
            chunks.append({"chunk_index": idx, "content": chunk})
        start = end - CHUNK_OVERLAP
        idx += 1
    return chunks

def get_metadata(pdf_path):
    name = pdf_path.stem.lower()
    board = "NCERT"
    for part in pdf_path.parts:
        if part.upper() in ("NCERT", "CBSE", "ICSE"):
            board = part.upper()
            break
    class_match = re.search(r"class[_\s]?(\d+)", name)
    class_level = class_match.group(1) if class_match else "10"
    subject = re.sub(r"class[_\s]?\d+[_\s]?", "", name).replace("_", " ").strip().title()
    if not subject:
        subject = "General"
    return board, class_level, subject

def main():
    if not all([GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
        print("ERROR: Missing env variables. Check your .env file.")
        return

    gemini   = genai.Client(api_key=GEMINI_API_KEY)
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    pdf_dir = Path("./textbooks")
    pdfs    = list(pdf_dir.rglob("*.pdf"))

    if not pdfs:
        print("No PDFs found in ./textbooks folder")
        return

    for pdf_path in pdfs:
        print(f"\nProcessing: {pdf_path.name}")
        board, class_level, subject = get_metadata(pdf_path)
        print(f"  Board={board}  Class={class_level}  Subject={subject}")

        text   = extract_text(pdf_path)
        chunks = make_chunks(text)
        print(f"  {len(chunks)} chunks created")

        texts      = [c["content"] for c in chunks]
        embeddings = []

        for i in range(0, len(texts), 100):
            batch  = texts[i:i + 100]
            result = gemini.models.embed_content(model=EMBED_MODEL, contents=batch)
            embeddings.extend([e.values for e in result.embeddings])
            time.sleep(0.5)
            print(f"  Embedded {min(i+100, len(texts))}/{len(texts)} chunks...")

        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            rows.append({
                "board":       board,
                "class_level": class_level,
                "subject":     subject,
                "source_file": pdf_path.name,
                "chunk_index": chunk["chunk_index"],
                "content":     chunk["content"],
                "embedding":   embedding,
            })

        for i in range(0, len(rows), 50):
            supabase.table("textbook_chunks").insert(rows[i:i + 50]).execute()

        print(f"  Uploaded {len(rows)} chunks to Supabase")

    print("\nDone! Check Supabase -> Table Editor -> textbook_chunks")

if __name__ == "__main__":
    main()
