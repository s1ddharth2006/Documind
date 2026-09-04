"""PDF and Document Ingestion Module using PyMuPDF.

Extracts text from PDF and TXT documents with robust error handling
for corrupted files, password-protected PDFs, empty pages, and scanned documents.
"""

from __future__ import annotations
import io
import os
from typing import Any, Dict, List, Optional
import pymupdf


class DocumentProcessingError(Exception):
    """Raised when document extraction encounters an unrecoverable error."""
    pass


class DocumentProcessor:
    """Production document ingestion processor supporting PDF and TXT formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
    MAX_RECOMMENDED_PAGES = 300  # Safety threshold to prevent OOM on massive documents

    @classmethod
    def process_file(
        cls,
        file_input: str | bytes | io.BytesIO,
        filename: str = "document",
    ) -> Dict[str, Any]:
        """Ingest and extract text from a file path, raw bytes, or BytesIO buffer.

        Args:
            file_input: File path string, raw bytes, or BytesIO buffer.
            filename: Name of the uploaded file for format inference.

        Returns:
            Dict containing:
                - raw_text (str): Full concatenated extracted text.
                - pages (List[Dict]): Per-page text and page index.
                - page_count (int): Number of pages.
                - file_type (str): 'pdf' or 'txt'.
                - file_name (str): Original file name.
                - metadata (dict): Author, title, creation date etc. (if PDF).
                - warnings (List[str]): User-facing warnings (e.g. image-only PDF).

        Raises:
            DocumentProcessingError: For corrupted, empty, or unsupported files.
        """
        ext = os.path.splitext(filename.lower())[1]
        if not ext:
            if isinstance(file_input, str):
                ext = os.path.splitext(file_input.lower())[1]

        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise DocumentProcessingError(
                f"Unsupported file format '{ext}'. Only PDF (.pdf) and Plain Text (.txt) files are supported."
            )

        if ext == ".pdf":
            return cls._extract_from_pdf(file_input, filename)
        else:
            return cls._extract_from_txt(file_input, filename)

    @classmethod
    def _extract_from_txt(
        cls,
        file_input: str | bytes | io.BytesIO,
        filename: str
    ) -> Dict[str, Any]:
        """Extract text from plain text file with auto encoding detection."""
        warnings: List[str] = []
        try:
            if isinstance(file_input, str):
                with open(file_input, "rb") as f:
                    content_bytes = f.read()
            elif isinstance(file_input, io.BytesIO):
                content_bytes = file_input.getvalue()
            elif isinstance(file_input, bytes):
                content_bytes = file_input
            else:
                content_bytes = bytes(file_input)
        except Exception as e:
            raise DocumentProcessingError(f"Failed to read TXT file: {str(e)}")

        if not content_bytes or len(content_bytes.strip()) == 0:
            raise DocumentProcessingError("The uploaded TXT file is empty (0 bytes).")

        # Decode using utf-8 with fallback to latin-1 and cp1252
        text = ""
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                text = content_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if not text:
            text = content_bytes.decode("utf-8", errors="replace")
            warnings.append("Notice: Some non-standard characters were replaced during encoding conversion.")

        if not text.strip():
            raise DocumentProcessingError("The TXT file contains only whitespace characters.")

        return {
            "raw_text": text,
            "pages": [{"page_number": 1, "text": text}],
            "page_count": 1,
            "file_type": "txt",
            "file_name": filename,
            "metadata": {"format": "Plain Text"},
            "warnings": warnings,
        }

    @classmethod
    def _extract_from_pdf(
        cls,
        file_input: str | bytes | io.BytesIO,
        filename: str
    ) -> Dict[str, Any]:
        """Extract text and metadata from PDF using PyMuPDF."""
        warnings: List[str] = []
        doc = None

        try:
            if isinstance(file_input, str):
                if not os.path.exists(file_input):
                    raise DocumentProcessingError(f"File not found: {file_input}")
                doc = pymupdf.open(file_input)
            elif isinstance(file_input, io.BytesIO):
                stream_bytes = file_input.getvalue()
                if not stream_bytes:
                    raise DocumentProcessingError("The uploaded PDF file is empty (0 bytes).")
                doc = pymupdf.open(stream=stream_bytes, filetype="pdf")
            elif isinstance(file_input, bytes):
                if not file_input:
                    raise DocumentProcessingError("The uploaded PDF file is empty (0 bytes).")
                doc = pymupdf.open(stream=file_input, filetype="pdf")
            else:
                doc = pymupdf.open(stream=file_input.read(), filetype="pdf")
        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(
                f"Corrupted or unreadable PDF document. Error details: {str(e)}"
            )

        if doc is None:
            raise DocumentProcessingError("Could not initialize PDF document stream.")

        if doc.is_encrypted:
            doc.close()
            raise DocumentProcessingError(
                "The PDF is encrypted or password-protected. Please provide an unlocked document."
            )

        total_pages = doc.page_count
        if total_pages == 0:
            doc.close()
            raise DocumentProcessingError("The PDF document contains 0 pages.")

        if total_pages > cls.MAX_RECOMMENDED_PAGES:
            warnings.append(
                f"Large document detected ({total_pages} pages). Processing first {cls.MAX_RECOMMENDED_PAGES} pages for optimal performance."
            )

        process_page_count = min(total_pages, cls.MAX_RECOMMENDED_PAGES)
        pages: List[Dict[str, Any]] = []
        extracted_text_chunks: List[str] = []

        total_chars = 0
        for i in range(process_page_count):
            page = doc.load_page(i)
            page_text = page.get_text("text") or ""
            page_text_clean = page_text.strip()
            pages.append({
                "page_number": i + 1,
                "text": page_text_clean,
                "char_count": len(page_text_clean),
            })
            if page_text_clean:
                extracted_text_chunks.append(page_text_clean)
                total_chars += len(page_text_clean)

        raw_text = "\n\n".join(extracted_text_chunks)

        # Detect scanned / image-only PDFs
        if total_chars < 30:
            warnings.append(
                "Very little or no selectable text found in the PDF. This document may be an image scan or contain flattened vector graphics."
            )

        # Extract metadata
        meta = doc.metadata or {}
        extracted_metadata = {
            "title": meta.get("title") or "Untitled",
            "author": meta.get("author") or "Unknown",
            "subject": meta.get("subject") or "",
            "producer": meta.get("producer") or "",
            "creation_date": meta.get("creationDate") or "",
        }

        doc.close()

        return {
            "raw_text": raw_text,
            "pages": pages,
            "page_count": total_pages,
            "file_type": "pdf",
            "file_name": filename,
            "metadata": extracted_metadata,
            "warnings": warnings,
        }
