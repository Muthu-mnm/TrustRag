"""
PDF extraction and page-aware chunking preserving metadata.
Preserves the repository's original PyPDFLoader while enhancing metadata tracking.
"""

import os
import tempfile
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def process_pdf_file(
    file_bytes: bytes,
    filename: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> Dict[str, Any]:
    """
    Process an uploaded PDF file:
    1. Writes to a secure temporary file.
    2. Uses PyPDFLoader to load pages with page number metadata.
    3. Splits pages into chunks while preserving page number, chunk ID, and source file metadata.
    4. Cleans up the temporary file safely.
    
    Returns a dict with:
        'documents': list of Document objects
        'page_count': total pages loaded
        'chunk_count': total chunks generated
        'has_text': whether selectable text was extracted
        'error': error message if failed
    """
    temp_file_path = None
    try:
        # Write to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            temp_file_path = tmp.name

        # Load using existing repository's PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        raw_pages = loader.load()

        if not raw_pages:
            return {
                "documents": [],
                "page_count": 0,
                "chunk_count": 0,
                "has_text": False,
                "error": "The uploaded PDF has 0 pages or could not be opened.",
            }

        # Check for empty or scanned PDF
        total_text_length = sum(len(doc.page_content.strip()) for doc in raw_pages)
        if total_text_length < 20:
            return {
                "documents": [],
                "page_count": len(raw_pages),
                "chunk_count": 0,
                "has_text": False,
                "error": "The uploaded PDF appears to be empty or contains scanned images without selectable text.",
            }

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        all_chunks: List[Document] = []
        chunk_idx = 0

        for page in raw_pages:
            # PyPDFLoader returns 0-indexed page in metadata 'page'
            page_num = page.metadata.get("page", 0) + 1
            source_doc = filename

            page_content = page.page_content.strip()
            if not page_content:
                continue

            page_splits = text_splitter.split_text(page_content)
            for split_idx, split_text in enumerate(page_splits):
                chunk_id = f"{source_doc}_p{page_num}_c{chunk_idx}"
                metadata = {
                    "source": source_doc,
                    "page": page_num,
                    "chunk_id": chunk_id,
                    "split_idx": split_idx,
                }
                all_chunks.append(Document(page_content=split_text, metadata=metadata))
                chunk_idx += 1

        return {
            "documents": all_chunks,
            "page_count": len(raw_pages),
            "chunk_count": len(all_chunks),
            "has_text": True,
            "error": None,
        }

    except Exception as e:
        return {
            "documents": [],
            "page_count": 0,
            "chunk_count": 0,
            "has_text": False,
            "error": f"Failed to extract text from PDF: {str(e)}",
        }
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
