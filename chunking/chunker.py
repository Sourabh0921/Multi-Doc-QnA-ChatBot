import re
import os
from typing import List, Dict
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
import uuid


def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def fixed_size_chunking(text: str, chunk_size: int = 512, overlap: int = 128) -> List[Dict]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        chunks.append({
            "content": chunk_text,
            "metadata": {
                "chunk_id": str(uuid.uuid4()),
                "strategy": "fixed",
                "start_word": start,
                "end_word": end
            }
        })
        start += chunk_size - overlap
        print("fixed_chunking--------------",chunks)

    return chunks


def semantic_chunking(text: str, level: str = "paragraph") -> List[Dict]:
    if level == "sentence":
        segments = sent_tokenize(text)
    else:  # paragraph
        segments = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for i, seg in enumerate(segments):
        chunks.append({
            "content": seg,
            "metadata": {
                "chunk_id": str(uuid.uuid4()),
                "strategy": "semantic",
                "segment_number": i
            }
        })
    # print("Semantic_chuncking--------------",chunks)
    return chunks


def custom_chunking(text: str) -> List[Dict]:
    """
    Custom chunking based on detecting section headings using regex.
    e.g., matches like "1. Introduction", "2.1 Engine Info", etc.
    """
    pattern = r'\n(?P<header>\d+(\.\d+)*\s+[^\n]+)\n'
    matches = list(re.finditer(pattern, text))
    chunks = []

    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "chunk_id": str(uuid.uuid4()),
                    "strategy": "custom",
                    "section_title": matches[i].group("header").strip()
                }
            })
    return chunks


def hierarchical_chunking(pdf_path: str) -> List[Dict]:
    text = read_pdf(pdf_path)
    lines = text.split('\n')

    chunks = []
    current_section = {"title": None, "content": "", "level": 0, "metadata": {}}

    section_pattern = re.compile(r"^(Chapter\s+\w+|[0-9IVXLCDM]{1,2}(\.[0-9]+)*\s+.+)", re.IGNORECASE)

    section_id = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = section_pattern.match(line)
        if match:
            # Save previous section
            if current_section["content"]:
                chunks.append({
                    "content": current_section["content"].strip(),
                    "metadata": {
                        "chunk_id": str(uuid.uuid4()),
                        "strategy": "hierarchical",
                        "section_title": current_section["title"],
                        "level": current_section["level"],
                        "section_id": section_id
                    }
                })
                section_id += 1

            # Start new section
            current_section["title"] = line
            current_section["content"] = ""
            current_section["level"] = line.count('.')  # Estimate depth based on dots
        else:
            current_section["content"] += line + " "

    # Add last section
    if current_section["content"]:
        chunks.append({
            "content": current_section["content"].strip(),
            "metadata": {
                "chunk_id": str(uuid.uuid4()),
                "strategy": "hierarchical",
                "section_title": current_section["title"],
                "level": current_section["level"],
                "section_id": section_id
            }
        })

    return chunks


def chunk_pdf_document(
    file_path: str,
    strategy: str = "fixed",
    chunk_size: int = 512,
    overlap: int = 128,
    semantic_level: str = "paragraph"
) -> List[Dict]:
    
    if strategy == "hierarchical":
        text = read_pdf(file_path)
        return hierarchical_chunking(text)
    
    text = read_pdf(file_path)
    
    if strategy == "fixed":
        return fixed_size_chunking(text, chunk_size=chunk_size, overlap=overlap)
    elif strategy == "semantic":
        return semantic_chunking(text, level=semantic_level)
    elif strategy == "custom":
        return custom_chunking(text)
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")
