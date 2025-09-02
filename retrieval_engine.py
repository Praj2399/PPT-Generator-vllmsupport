#!/usr/bin/env python3
"""
Retrieval and context extraction functionality with hybrid search
- OCR-aware ingestion via DocumentProcessor (PDFs/DOCX/TXT; OCR on weak pages)
- Vectorstore caching + optional persistence
- MMR-based dense retriever + BM25 via EnsembleRetriever
- Retrieval scales with requested slide count (desired_slides)
- Adaptive sufficiency gate (thresholds scale with ask size)
"""

import os
import re
import hashlib
from typing import List, Optional, Any, Dict, Tuple, Set

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from .model_factory import ModelFactory
from .config import (
    RETRIEVAL_K,
    MIN_WORDS_THRESHOLD,
    MIN_STRUCTURE_LINES,
    PATTERNS,
)
from .document_processor import DocumentProcessor


class RetrievalEngine:
    """Handles vector store operations and context extraction"""

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Args:
            persist_directory: if provided, Chroma will persist embeddings to this path.
                               If None, vectorstore is ephemeral per process.
        """
        self.embeddings = ModelFactory.create_embeddings()
        self.document_processor = DocumentProcessor()
        self.persist_directory = persist_directory
        self._vs_cache: Dict[str, Tuple[Chroma, List]] = {}  # key -> (vectorstore, splits)

    # -----------------------
    # Internal: caching utils
    # -----------------------
    def _fingerprint_files(self, file_paths: List[str]) -> str:
        """Create a stable fingerprint from file paths + size + mtime (fast; no file reads)."""
        h = hashlib.sha256()
        for p in sorted(set(fp for fp in file_paths if fp)):
            try:
                st = os.stat(p)
                h.update(p.encode("utf-8"))
                h.update(str(st.st_size).encode("utf-8"))
                h.update(str(int(st.st_mtime)).encode("utf-8"))
            except Exception:
                # If stat fails, still include the path string so cache key changes if user edits list
                h.update(p.encode("utf-8"))
        return h.hexdigest()

    def _get_vectorstore(self, file_paths: List[str]) -> Optional[Tuple[Chroma, List]]:
        """
        Build (or reuse cached) vectorstore + splits from files using OCR-aware processor.
        Returns (vectorstore, splits) or None on failure.
        """
        if not file_paths:
            return None

        cache_key = self._fingerprint_files(file_paths)
        if cache_key in self._vs_cache:
            return self._vs_cache[cache_key]

        # Load + split via DocumentProcessor (has OCR fallback for weak PDF pages)
        splits = self.document_processor.load_and_process_files(file_paths)
        if not splits:
            return None

        # Create vectorstore (persist if directory provided)
        try:
            if self.persist_directory:
                os.makedirs(self.persist_directory, exist_ok=True)
                vs = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                )
                vs.persist()
            else:
                vs = Chroma.from_documents(documents=splits, embedding=self.embeddings)
        except Exception as e:
            print(f"[retrieval] Failed to create vectorstore: {e}")
            return None

        self._vs_cache[cache_key] = (vs, splits)
        return vs, splits

    # -----------------------
    # Public: simple semantic retriever (MMR)
    # -----------------------
    def create_retriever(self, file_paths: List[str], desired_slides: Optional[int] = None) -> Optional[Any]:
        """
        Create a semantic retriever (MMR) from file paths.
        Scales fetch_k with desired_slides for better diversity on big decks.
        """
        vs_and_splits = self._get_vectorstore(file_paths)
        if not vs_and_splits:
            return None
        vectorstore, _ = vs_and_splits

        # Scale MMR fetch_k for bigger decks (more candidates -> more diversity)
        big = bool(desired_slides and desired_slides >= 20)
        scale = 5 if big else 3
        fetch_k = max(RETRIEVAL_K * scale, 30 if big else 10)

        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RETRIEVAL_K, "fetch_k": fetch_k, "lambda_mult": 0.5},
        )

    def extract_document_context(self, file_paths: List[str], topic: str, desired_slides: Optional[int] = None) -> str:
        """Extract context from documents for a given topic (semantic MMR; scaled)."""
        if not file_paths:
            return ""

        retriever = self.create_retriever(file_paths, desired_slides=desired_slides)
        if not retriever:
            return ""

        try:
            # Handle both Runnable and legacy APIs
            try:
                docs = retriever.invoke(topic)
            except AttributeError:
                docs = retriever.get_relevant_documents(topic)
            return "\n\n".join(doc.page_content for doc in docs)
        except Exception:
            return ""

    # -----------------------
    # Public: hybrid retriever
    # -----------------------
    def create_hybrid_retriever(
        self, file_paths: List[str], topic: str = "", desired_slides: Optional[int] = None
    ) -> Optional[Any]:
        """
        Create hybrid retriever combining semantic MMR and BM25 keyword search.
        Scales MMR fetch_k with desired_slides.
        """
        vs_and_splits = self._get_vectorstore(file_paths)
        if not vs_and_splits:
            print("❌ No documents loaded successfully")
            return None
        vectorstore, splits = vs_and_splits

        big = bool(desired_slides and desired_slides >= 20)
        scale = 5 if big else 3
        fetch_k = max(RETRIEVAL_K * scale, 30 if big else 10)

        try:
            semantic_retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": RETRIEVAL_K, "fetch_k": fetch_k, "lambda_mult": 0.5},
            )
            keyword_retriever = BM25Retriever.from_documents(splits, k=RETRIEVAL_K)

            hybrid_retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, keyword_retriever],
                weights=[0.6, 0.4],
            )
            print("✅ Hybrid retriever created (60% semantic + 40% keyword)")
            return hybrid_retriever

        except Exception as e:
            print(f"⚠️ Hybrid retrieval failed, falling back to semantic: {e}")
            return vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": RETRIEVAL_K, "fetch_k": fetch_k, "lambda_mult": 0.5},
            )

    # -----------------------
    # Context sufficiency (adaptive)
    # -----------------------
    def analyze_content_sufficiency(self, context: str, desired_slides: Optional[int] = None) -> bool:
        """
        Analyze if context is sufficient for document-based generation.
        Thresholds scale with desired_slides (e.g., ×3 for 30 slides).
        """
        if not context:
            return False

        words = context.split()
        lines = context.split("\n")

        scale = 1 if not desired_slides else max(1, desired_slides // 10)  # e.g., 30 → 3x thresholds
        has_enough_words = len(words) > (MIN_WORDS_THRESHOLD * scale)
        has_structure = len(lines) > (MIN_STRUCTURE_LINES * scale)

        # Data detection using patterns from config (robust to missing keys)
        has_numbers = bool(re.search(PATTERNS.get("numbers", r"\b\d+(?:\.\d+)?\b"), context))
        has_percentages = bool(re.search(PATTERNS.get("percentages", r"\b\d+(?:\.\d+)?\s?%\b"), context))
        has_years = bool(re.search(PATTERNS.get("years", r"\b(19|20)\d{2}\b"), context))
        has_technical_terms = bool(
            re.search(PATTERNS.get("technical_terms", r"accuracy|dataset|method|results|evaluation"), context, re.IGNORECASE)
        )

        has_data = has_numbers or has_percentages or has_years or has_technical_terms
        return has_enough_words and has_structure and has_data

    # -----------------------
    # Enhanced multi-query ctx
    # -----------------------
    def generate_multiple_queries(self, main_topic: str, slide_topic: str) -> List[str]:
        """Generate single optimized query for fast retrieval."""
        # Use just one well-crafted query instead of 5 variations for performance
        return [f"{main_topic} {slide_topic}"]

    def _doc_key(self, doc) -> str:
        """Stable dedup key based on source and page; falls back to content hash."""
        meta = getattr(doc, "metadata", {}) or {}
        src = str(meta.get("source", ""))
        pn = meta.get("page_number", meta.get("page"))
        if src or pn is not None:
            return f"{src}|{pn}"
        return f"content|{hash(doc.page_content[:512])}"

    def enhanced_extract_document_context(
        self,
        file_paths: List[str],
        main_topic: str,
        slide_topic: str,
        desired_slides: Optional[int] = None,
    ) -> str:
        """
        Enhanced context extraction with hybrid retrieval and multi-query.
        - Hybrid: semantic MMR + BM25
        - Multi-query expansion based on slide intent
        - Dedup by (source, page_number)
        - Scales query budget and context cap by desired_slides
        """
        if not file_paths:
            return ""

        print(f"🔍 Enhanced retrieval for: {slide_topic}")

        hybrid_retriever = self.create_hybrid_retriever(file_paths, main_topic, desired_slides=desired_slides)
        if not hybrid_retriever:
            return ""

        try:
            queries = self.generate_multiple_queries(main_topic, slide_topic)

            # Scale query budget (min 5, grows slowly; cap 8)
            max_q = 5 if desired_slides is None else min(8, max(5, desired_slides // 6))
            queries = queries[:max_q]
            print(f"📝 Generated {len(queries)} queries: {queries}")

            all_chunks: List[str] = []
            seen: Set[str] = set()

            for i, q in enumerate(queries, 1):
                try:
                    print(f"   Query {i}: {q}")
                    try:
                        retrieved = hybrid_retriever.invoke(q)  # Runnable API
                    except AttributeError:
                        retrieved = hybrid_retriever.get_relevant_documents(q)  # legacy

                    for d in retrieved:
                        key = self._doc_key(d)
                        if key not in seen:
                            all_chunks.append(d.page_content)
                            seen.add(key)
                except Exception as qe:
                    print(f"⚠️ Query failed: {q} - {qe}")
                    continue

            print(f"📊 Retrieved {len(all_chunks)} unique chunks")

            # Scale context cap (~1 chunk per 2 slides; min 12; max 40)
            cap = 12 if desired_slides is None else min(40, max(12, desired_slides // 2))
            return "\n\n".join(all_chunks[:cap])

        except Exception as e:
            print(f"⚠️ Enhanced retrieval failed, falling back: {e}")
            return self.extract_document_context(file_paths, slide_topic, desired_slides=desired_slides)

    # -----------------------
    # Misc helpers (optional)
    # -----------------------
    def clean_text(self, text: str) -> str:
        """Clean text content (from document_processor)"""
        if not text:
            return ""
        text = " ".join(text.split())
        text = re.sub(r"[^\w\s\.\,\!\?\-\:\;\(\)]", "", text)
        return text
