#!/usr/bin/env python3
"""
Document processing functionality for PPT Generator
- Adds OCR fallback for scanned/low-text PDFs (PyMuPDF + PIL + pytesseract)
- Keeps previous API and behavior intact
"""
from typing import List, Optional, Iterable
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# Prefer the modern splitter; fall back if not installed
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from .config import CHUNK_SIZE, CHUNK_OVERLAP

# ---------- Tunables ----------
_MIN_TEXT_PER_PAGE_CHARS = 40           # if a page has less chars than this → OCR that page
_OCR_DEFAULT_LANG = "eng"               # Tesseract language; extend if you need multilingual OCR
_OCR_DPI = 200                          # rasterization DPI (balance speed/accuracy)
# --------------------------------


class DocumentProcessor:
    """Handles all document loading and processing operations (with OCR fallback)"""

    def __init__(self, ocr_lang: str = _OCR_DEFAULT_LANG, ocr_dpi: int = _OCR_DPI):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.ocr_lang = ocr_lang
        self.ocr_dpi = ocr_dpi

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text to handle encoding/null-char issues"""
        if not text:
            return ""
        # strip nulls and weird control chars, keep it simple & fast
        text = text.replace("\x00", "")
        return text.encode("utf-8", "ignore").decode("utf-8")

    # ---------- Internal helpers ----------

    def _ocr_pdf_pages(self, file_path: str, page_indexes: Optional[Iterable[int]] = None) -> List[Document]:
        """
        OCR specific pages from a PDF using PyMuPDF + Tesseract.
        Returns a list of Document(page_content, metadata={source,page_number,ocr=True})
        """
        try:
            import fitz  # PyMuPDF
            from PIL import Image
            import pytesseract
        except Exception as e:
            # If OCR stack isn't available, return empty → caller will keep original text
            print(f"[OCR disabled] Missing dependency: {e}")
            return []

        docs: List[Document] = []
        with fitz.open(file_path) as pdf:
            indexes = list(page_indexes) if page_indexes is not None else list(range(len(pdf)))
            # safety: bound indices
            indexes = [i for i in indexes if 0 <= i < len(pdf)]

            for i in indexes:
                page = pdf[i]
                # Rasterize page
                zoom = self.ocr_dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # OCR
                text = pytesseract.image_to_string(img, lang=self.ocr_lang)
                text = self.clean_text(text)
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": file_path, "page_number": i + 1, "ocr": True},
                        )
                    )
        return docs

    def _load_pdf_with_ocr_fallback(self, file_path: str) -> List[Document]:
        """
        Load PDF pages via PyPDFLoader; OCR only those pages that came back too short.
        """
        base_docs = PyPDFLoader(file_path).load()  # usually returns one Document per page
        if not base_docs:
            # fully scanned? try OCR all pages
            return self._ocr_pdf_pages(file_path)

        # sanitize + normalize metadata
        page_texts = []
        pages_to_ocr = []
        for i, d in enumerate(base_docs):
            # clean text
            d.page_content = self.clean_text(d.page_content or "")
            # normalize page number metadata
            # PyPDFLoader puts 'page' (0-based) in metadata; we also add 1-based 'page_number'
            page_num = (d.metadata.get("page") or i) + 1
            d.metadata = {**d.metadata, "source": file_path, "page_number": page_num, "ocr": False}

            page_texts.append(d.page_content)
            if len(d.page_content.strip()) < _MIN_TEXT_PER_PAGE_CHARS:
                pages_to_ocr.append(i)  # 0-based index for this page

        if not pages_to_ocr:
            return base_docs  # all good, no OCR needed

        # OCR only the weak pages
        ocr_docs = self._ocr_pdf_pages(file_path, page_indexes=pages_to_ocr)
        if not ocr_docs:
            # OCR unavailable/failed: return original docs
            return base_docs

        # Merge: replace weak pages with their OCR text (by page_number)
        ocr_map = {d.metadata["page_number"]: d for d in ocr_docs}
        merged: List[Document] = []
        for d in base_docs:
            pn = d.metadata.get("page_number")
            merged.append(ocr_map.get(pn, d))
        return merged

    # ---------- Public API ----------

    def load_and_process_files(self, file_paths: List[str]):
        """Optimized file loading and processing with OCR fallback for PDFs"""
        if not file_paths or not any(file_paths):
            return None

        all_docs: List[Document] = []
        for file_path in file_paths:
            file_path = (file_path or "").strip()
            if not file_path:
                continue

            try:
                if file_path.lower().endswith(".pdf"):
                    documents = self._load_pdf_with_ocr_fallback(file_path)
                elif file_path.lower().endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    documents = loader.load()
                elif file_path.lower().endswith(".txt"):
                    loader = TextLoader(file_path)
                    documents = loader.load()
                else:
                    # unsupported extension: skip silently (consistent with your original code)
                    continue

                # sanitize content for non-PDF loaders as well
                for doc in documents:
                    doc.page_content = self.clean_text(doc.page_content)
                    # ensure minimal metadata consistency
                    if "source" not in doc.metadata:
                        doc.metadata["source"] = file_path
                    if "page_number" not in doc.metadata:
                        # treat non-paged docs as page 1
                        doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1
                    if "ocr" not in doc.metadata:
                        doc.metadata["ocr"] = False

                all_docs.extend(documents)
            except Exception as e:
                # match original behavior: skip failed files, but log once
                print(f"[load warning] Skipped '{file_path}': {e}")
                continue

        if not all_docs:
            return None

        # Chunk
        splits = self.text_splitter.split_documents(all_docs)
        return splits
