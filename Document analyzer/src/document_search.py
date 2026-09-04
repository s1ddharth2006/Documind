"""Document Search Engine Module.

Implements vector space retrieval using TF-IDF and Cosine Similarity
to find and rank relevant paragraphs/sections against user queries,
with term highlighting and page attribution.
"""

from __future__ import annotations
import html
import re
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.text_preprocessor import DEFAULT_STOP_WORDS, TextPreprocessor


class DocumentSearchEngine:
    """TF-IDF and Cosine Similarity vector space search engine."""

    def __init__(self, min_similarity_threshold: float = 0.05) -> None:
        self.min_similarity_threshold = min_similarity_threshold
        self.preprocessor = TextPreprocessor()
        self.sections: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

    def index_document(
        self,
        raw_text: str,
        pages: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Index document sections/paragraphs into a TF-IDF vector space.

        Args:
            raw_text: Concatenated full document text.
            pages: Optional list of per-page text dictionaries for page tracking.

        Returns:
            Number of indexed sections.
        """
        self.sections = []

        # If pages are provided, segment each page into paragraphs with page attribution
        if pages and len(pages) > 0:
            section_idx = 1
            for page_info in pages:
                page_num = page_info.get("page_number", 1)
                page_text = page_info.get("text", "").strip()
                if not page_text:
                    continue

                paragraphs = self.preprocessor.split_into_paragraphs(page_text)
                if not paragraphs:
                    paragraphs = [page_text]

                for p in paragraphs:
                    if len(p.strip()) > 15:
                        self.sections.append({
                            "section_id": section_idx,
                            "page_number": page_num,
                            "text": p.strip(),
                            "word_count": len(self.preprocessor.tokenize_words(p)),
                        })
                        section_idx += 1
        else:
            # Fallback to general paragraph segmentation
            paragraphs = self.preprocessor.split_into_paragraphs(raw_text)
            for idx, p in enumerate(paragraphs, start=1):
                if len(p.strip()) > 15:
                    self.sections.append({
                        "section_id": idx,
                        "page_number": 1,
                        "text": p.strip(),
                        "word_count": len(self.preprocessor.tokenize_words(p)),
                    })

        if not self.sections:
            # If no paragraphs met length, index the entire text as a single section
            cleaned = self.preprocessor.clean_text(raw_text)
            if cleaned:
                self.sections.append({
                    "section_id": 1,
                    "page_number": 1,
                    "text": cleaned,
                    "word_count": len(self.preprocessor.tokenize_words(cleaned)),
                })

        if not self.sections:
            return 0

        # Fit TF-IDF Vectorizer across document sections
        corpus = [s["text"] for s in self.sections]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=list(DEFAULT_STOP_WORDS),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[a-zA-Z0-9]{2,}\b",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        return len(self.sections)

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search the indexed sections using TF-IDF query vector and Cosine Similarity.

        Args:
            query: User search string or question.
            top_k: Maximum number of results to return.
            threshold: Minimum cosine similarity score (0.0 - 1.0).

        Returns:
            List of matching sections ranked by similarity score:
                [
                    {
                        "section_id": int,
                        "page_number": int,
                        "score": float (cosine similarity 0.0 - 1.0),
                        "score_percent": float (0.0 - 100.0),
                        "highlighted_text": str (HTML snippet with <mark> tags),
                        "raw_text": str,
                        "matched_terms": List[str]
                    }, ...
                ]
        """
        clean_query = (query or "").strip()
        min_thresh = threshold if threshold is not None else self.min_similarity_threshold

        if not clean_query or self.vectorizer is None or self.tfidf_matrix is None or not self.sections:
            return []

        # Vectorize query in the document's vocabulary space
        query_vec = self.vectorizer.transform([clean_query])

        if query_vec.nnz == 0:
            # Query words are out of vocabulary; do a fallback token match
            return self._fallback_keyword_search(clean_query, top_k)

        # Compute cosine similarity between query vector and all section vectors
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Filter and rank
        matched_indices = np.where(similarities >= min_thresh)[0]
        if len(matched_indices) == 0:
            # If strict threshold yielded 0, take highest non-zero if above 0.01
            non_zero = np.where(similarities > 0.005)[0]
            if len(non_zero) > 0:
                matched_indices = non_zero

        ranked_indices = matched_indices[np.argsort(-similarities[matched_indices])][:top_k]

        # Extract query tokens for highlighting
        query_tokens = [
            t.lower() for t in self.preprocessor.tokenize_words(clean_query)
            if t.lower() not in DEFAULT_STOP_WORDS and len(t) > 1
        ]
        if not query_tokens:
            query_tokens = [t.lower() for t in self.preprocessor.tokenize_words(clean_query)]

        results: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            sec = self.sections[idx]
            highlighted, terms = self._highlight_matches(sec["text"], query_tokens)

            results.append({
                "section_id": sec["section_id"],
                "page_number": sec["page_number"],
                "score": round(score, 4),
                "score_percent": round(score * 100, 1),
                "highlighted_text": highlighted,
                "raw_text": sec["text"],
                "matched_terms": terms,
                "word_count": sec["word_count"],
            })

        return results

    def _fallback_keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback substring / word search if TF-IDF vector is zero (e.g. unique numbers or rare tokens)."""
        tokens = [t.lower() for t in self.preprocessor.tokenize_words(query) if len(t) > 1]
        if not tokens:
            return []

        scored_sections = []
        for sec in self.sections:
            text_lower = sec["text"].lower()
            matches = sum(1 for t in tokens if t in text_lower)
            if matches > 0:
                score = matches / len(tokens)
                highlighted, matched_terms = self._highlight_matches(sec["text"], tokens)
                scored_sections.append({
                    "section_id": sec["section_id"],
                    "page_number": sec["page_number"],
                    "score": round(score * 0.5, 4),  # Lower confidence scale for fallback
                    "score_percent": round(score * 50, 1),
                    "highlighted_text": highlighted,
                    "raw_text": sec["text"],
                    "matched_terms": matched_terms,
                    "word_count": sec["word_count"],
                })

        scored_sections.sort(key=lambda x: x["score"], reverse=True)
        return scored_sections[:top_k]

    @staticmethod
    def _highlight_matches(text: str, query_tokens: List[str]) -> tuple[str, List[str]]:
        """Highlight query tokens inside text with HTML <mark> tags."""
        if not query_tokens:
            return html.escape(text), []

        matched_terms = set()
        escaped_tokens = [re.escape(t) for t in query_tokens if t]
        if not escaped_tokens:
            return html.escape(text), []

        pattern = re.compile(r"\b(" + "|".join(escaped_tokens) + r")\b", re.IGNORECASE)

        def replace_match(match: re.Match) -> str:
            val = match.group(0)
            matched_terms.add(val.lower())
            return f'<mark class="search-match" style="background-color: #fde047; color: #1e293b; padding: 0.1rem 0.35rem; border-radius: 4px; font-weight: 600;">{html.escape(val)}</mark>'

        highlighted = pattern.sub(replace_match, text)
        return highlighted, sorted(list(matched_terms))
