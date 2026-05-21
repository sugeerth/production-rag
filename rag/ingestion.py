"""Document ingestion: parsing, cleaning, and chunking."""

import os
import re
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import config


@dataclass
class DocumentChunk:
    """A chunk of a document with metadata."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            **self.metadata,
        }


@dataclass
class IngestStats:
    """Statistics from an ingestion run."""
    total_docs: int = 0
    total_chunks: int = 0
    failed_docs: int = 0
    avg_chunk_size: float = 0.0
    errors: list = field(default_factory=list)


def generate_doc_id(filepath: str) -> str:
    """Generate a stable document ID from file path."""
    return hashlib.md5(filepath.encode()).hexdigest()


def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate a stable chunk ID."""
    return hashlib.md5(f"{doc_id}:{chunk_index}".encode()).hexdigest()


# --- Parsers ---

def parse_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_markdown(filepath: str) -> str:
    return parse_text(filepath)


def parse_pdf(filepath: str) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(f"[Page {i+1}]\n{text}")
        return "\n\n".join(pages)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF {filepath}: {e}")


def parse_docx(filepath: str) -> str:
    try:
        from docx import Document
        doc = Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        raise ValueError(f"Failed to parse DOCX {filepath}: {e}")


def parse_html(filepath: str) -> str:
    try:
        from bs4 import BeautifulSoup
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        raise ValueError(f"Failed to parse HTML {filepath}: {e}")


PARSERS = {
    ".txt": parse_text,
    ".md": parse_markdown,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".html": parse_html,
    ".htm": parse_html,
}


def parse_document(filepath: str) -> str:
    """Parse a document file into plain text."""
    ext = os.path.splitext(filepath)[1].lower()
    parser = PARSERS.get(ext)
    if not parser:
        raise ValueError(f"Unsupported file format: {ext}")
    return parser(filepath)


# --- Cleaning ---

def clean_text(text: str) -> str:
    """Clean extracted text: normalize whitespace, remove artifacts."""
    # Normalize unicode whitespace
    text = re.sub(r"\r\n", "\n", text)
    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Remove null bytes and control chars (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


# --- Chunking ---

def _split_by_headings(text: str) -> list[dict]:
    """Split markdown/text by headings, preserving heading hierarchy."""
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    sections = []
    last_end = 0
    current_heading = ""

    for match in heading_pattern.finditer(text):
        if last_end < match.start():
            content = text[last_end:match.start()].strip()
            if content:
                sections.append({
                    "heading": current_heading,
                    "content": content
                })
        current_heading = match.group(2).strip()
        last_end = match.end()

    # Remaining text
    remaining = text[last_end:].strip()
    if remaining:
        sections.append({
            "heading": current_heading,
            "content": remaining
        })

    if not sections:
        sections.append({"heading": "", "content": text})

    return sections


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None,
               min_chunk_size: int = None) -> list[str]:
    """Chunk text into overlapping segments.

    Uses a hybrid approach:
    1. First split by headings/sections (semantic boundaries)
    2. Then split large sections by sentences with overlap
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    min_chunk_size = min_chunk_size or config.MIN_CHUNK_SIZE

    sections = _split_by_headings(text)
    chunks = []

    # Merge small consecutive sections to avoid tiny chunks
    merged_sections = []
    for section in sections:
        if merged_sections and len(merged_sections[-1]["content"].split()) + len(section["content"].split()) <= chunk_size:
            prev = merged_sections[-1]
            heading_prefix = f"\n\n{section['heading']}\n\n" if section["heading"] else "\n\n"
            prev["content"] += heading_prefix + section["content"]
        else:
            merged_sections.append(dict(section))

    for section in merged_sections:
        heading = section["heading"]
        content = section["content"]
        prefix = f"{heading}\n\n" if heading else ""
        full_text = prefix + content
        words = full_text.split()

        if len(words) <= chunk_size:
            if len(words) >= min_chunk_size:
                chunks.append(full_text.strip())
        else:
            # Split by sentences for better boundaries
            sentences = re.split(r'(?<=[.!?])\s+', content)
            current_chunk_words = list(prefix.split()) if prefix else []
            current_chunk_parts = [prefix] if prefix else []

            for sentence in sentences:
                sentence_words = sentence.split()
                if len(current_chunk_words) + len(sentence_words) > chunk_size and len(current_chunk_words) >= min_chunk_size:
                    chunks.append(" ".join(current_chunk_parts).strip())
                    # Overlap: keep last N words
                    overlap_text = " ".join(current_chunk_words[-chunk_overlap:])
                    current_chunk_parts = [overlap_text, sentence]
                    current_chunk_words = overlap_text.split() + sentence_words
                else:
                    current_chunk_parts.append(sentence)
                    current_chunk_words.extend(sentence_words)

            if current_chunk_words and len(current_chunk_words) >= min_chunk_size:
                chunks.append(" ".join(current_chunk_parts).strip())

    # Fallback: if no chunks produced, just split by words
    if not chunks:
        words = text.split()
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.split()) >= min_chunk_size:
                chunks.append(chunk)

    return chunks


def ingest_document(filepath: str) -> list[DocumentChunk]:
    """Full pipeline: parse, clean, chunk a single document."""
    doc_id = generate_doc_id(filepath)
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    mod_time = datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()

    raw_text = parse_document(filepath)
    cleaned = clean_text(raw_text)
    text_chunks = chunk_text(cleaned)

    doc_chunks = []
    for i, chunk_text_content in enumerate(text_chunks):
        chunk = DocumentChunk(
            chunk_id=generate_chunk_id(doc_id, i),
            doc_id=doc_id,
            text=chunk_text_content,
            metadata={
                "source": filename,
                "filepath": filepath,
                "file_type": ext,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "modified": mod_time,
                "word_count": len(chunk_text_content.split()),
            }
        )
        doc_chunks.append(chunk)

    return doc_chunks


def ingest_directory(directory: str = None) -> tuple[list[DocumentChunk], IngestStats]:
    """Ingest all supported documents from a directory."""
    directory = directory or config.DOCUMENTS_DIR
    stats = IngestStats()
    all_chunks = []

    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
        return all_chunks, stats

    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in PARSERS:
                continue
            filepath = os.path.join(root, filename)
            stats.total_docs += 1
            try:
                chunks = ingest_document(filepath)
                all_chunks.extend(chunks)
            except Exception as e:
                stats.failed_docs += 1
                stats.errors.append({"file": filename, "error": str(e)})

    stats.total_chunks = len(all_chunks)
    if all_chunks:
        stats.avg_chunk_size = sum(c.metadata["word_count"] for c in all_chunks) / len(all_chunks)

    return all_chunks, stats
