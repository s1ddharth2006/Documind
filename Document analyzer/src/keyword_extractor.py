"""Keyword Extraction Module based on TF-IDF.

Extracts top salient keywords and keyphrases (unigrams and bigrams)
using mathematical TF-IDF scoring and term frequency analysis.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.text_preprocessor import DEFAULT_STOP_WORDS, TextPreprocessor


class TfidfKeywordExtractor:
    """Extracts top keywords and keyphrases from documents using TF-IDF."""

    def __init__(
        self,
        top_n: int = 15,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
    ) -> None:
        self.top_n = top_n
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.preprocessor = TextPreprocessor()

    def extract_keywords(
        self,
        text: str,
        top_n: Optional[int] = None,
        context_corpus: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract top N keywords with their TF-IDF scores and term frequencies.

        Args:
            text: Document text to extract keywords from.
            top_n: Override default number of keywords to return.
            context_corpus: Optional list of background documents to compute true IDF.
                           If None, text is split into paragraphs as local corpus.

        Returns:
            List of Dicts:
                [
                    {
                        "keyword": str,
                        "score": float (0.0 - 1.0 normalized TF-IDF),
                        "tfidf_raw": float,
                        "frequency": int,
                        "ngram_type": "unigram" | "bigram",
                        "rank": int
                    }, ...
                ]
        """
        n_results = top_n or self.top_n
        cleaned_text = self.preprocessor.clean_text(text)

        if not cleaned_text or len(cleaned_text.split()) < 3:
            return []

        # Build corpus for IDF calculation:
        # If no external corpus is supplied, treat document paragraphs as the collection.
        # This yields high IDF for terms distinctive to particular sections.
        if context_corpus and len(context_corpus) > 1:
            corpus = context_corpus
            target_doc = cleaned_text
        else:
            paragraphs = self.preprocessor.split_into_paragraphs(cleaned_text)
            if len(paragraphs) >= 2:
                corpus = paragraphs
                target_doc = cleaned_text
            else:
                sentences = self.preprocessor.split_into_sentences(cleaned_text)
                if len(sentences) >= 2:
                    corpus = sentences
                    target_doc = cleaned_text
                else:
                    # Single short block: duplicate with slight variation for IDF weighting
                    corpus = [cleaned_text, cleaned_text]
                    target_doc = cleaned_text

        # Vectorize using TF-IDF
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                stop_words=list(DEFAULT_STOP_WORDS),
                sublinear_tf=True,
                token_pattern=r"(?u)\b[a-zA-Z]{3,}(?:-[a-zA-Z]{3,})*\b",
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = np.array(vectorizer.get_feature_names_out())

            # Transform the full document
            doc_vector = vectorizer.transform([target_doc]).toarray().flatten()

            if doc_vector.sum() == 0:
                return []

            # Filter out non-zero indices and sort descending by TF-IDF score
            non_zero_indices = np.where(doc_vector > 0)[0]
            sorted_indices = non_zero_indices[np.argsort(-doc_vector[non_zero_indices])]

            # Calculate raw term occurrences in document for context
            lower_doc = target_doc.lower()
            results: List[Dict[str, Any]] = []

            max_score = float(doc_vector[sorted_indices[0]]) if len(sorted_indices) > 0 else 1.0

            rank = 1
            for idx in sorted_indices:
                keyword = str(feature_names[idx])
                raw_score = float(doc_vector[idx])
                normalized_score = round(raw_score / max_score, 4) if max_score > 0 else 0.0

                # Count term occurrences in text (word-boundary regex)
                escaped_kw = re.escape(keyword)
                freq = len(re.findall(r"\b" + escaped_kw + r"\b", lower_doc))

                ngram_type = "bigram" if " " in keyword else "unigram"

                results.append({
                    "keyword": keyword,
                    "score": normalized_score,
                    "tfidf_raw": round(raw_score, 4),
                    "frequency": max(1, freq),
                    "ngram_type": ngram_type,
                    "rank": rank,
                })

                rank += 1
                if len(results) >= n_results:
                    break

            return results

        except Exception as e:
            print(f"[WARNING] Keyword extraction fallback triggered: {e}")
            # Fallback to pure word frequency if TF-IDF fails (e.g. extremely short text)
            words = self.preprocessor.tokenize_words(cleaned_text, lower=True)
            filtered = [w for w in words if w not in DEFAULT_STOP_WORDS and len(w) > 3]
            from collections import Counter
            counts = Counter(filtered).most_common(n_results)
            max_c = counts[0][1] if counts else 1
            return [
                {
                    "keyword": word,
                    "score": round(count / max_c, 4),
                    "tfidf_raw": round(count / max_c, 4),
                    "frequency": count,
                    "ngram_type": "unigram",
                    "rank": idx + 1,
                }
                for idx, (word, count) in enumerate(counts)
            ]
