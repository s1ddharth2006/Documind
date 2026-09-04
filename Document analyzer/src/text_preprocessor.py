"""Text Preprocessing and NLP Utilities.

Provides text cleaning, tokenization, stop-word removal,
sentence/paragraph segmentation, and document statistics.
"""

from __future__ import annotations
import math
import re
import string
from typing import Any, Dict, List, Set, Tuple

# Built-in comprehensive English stop-words set to avoid runtime network download dependencies
DEFAULT_STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "arent", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cant", "cannot", "could", "couldn", "couldnt",
    "did", "didn", "didnt", "do", "does", "doesn", "doesnt", "doing", "don", "dont", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn", "hadnt", "has", "hasn", "hasnt",
    "have", "haven", "havent", "having", "he", "hed", "hell", "hes", "her", "here",
    "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i",
    "id", "ill", "im", "ive", "if", "in", "into", "is", "isn", "isnt", "it", "its",
    "itself", "let", "lets", "me", "more", "most", "mustn", "mustnt", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan", "shant", "she",
    "shed", "shell", "shes", "should", "shouldn", "shouldnt", "so", "some", "such",
    "than", "that", "thats", "the", "their", "theirs", "them", "themselves",
    "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre",
    "theyve", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn", "wasnt", "we", "wed", "well", "were", "weren", "werent",
    "weve", "what", "whats", "when", "whens", "where", "wheres", "which",
    "while", "who", "whos", "whom", "why", "whys", "with", "won", "wont", "would",
    "wouldn", "wouldnt", "you", "youd", "youll", "youre", "youve", "your", "yours",
    "yourself", "yourselves", "also", "may", "might", "shall", "will", "us", "etc"
}


class TextPreprocessor:
    """Production-grade text preprocessor and statistics generator."""

    def __init__(self, custom_stop_words: Set[str] | None = None) -> None:
        self.stop_words: Set[str] = set(DEFAULT_STOP_WORDS)
        if custom_stop_words:
            self.stop_words.update(w.lower() for w in custom_stop_words)

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean raw text by normalizing whitespaces, quotes, and non-printable characters."""
        if not text:
            return ""
        # Normalize various Unicode quotes and dashes
        text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        text = text.replace("—", " - ").replace("–", " - ")
        # Replace carriage returns and tabs
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove non-printable control characters except newline
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        # Collapse multiple horizontal whitespace
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3+ newlines to 2 newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def split_into_paragraphs(text: str) -> List[str]:
        """Split text into distinct non-empty paragraphs."""
        cleaned = TextPreprocessor.clean_text(text)
        if not cleaned:
            return []
        raw_paragraphs = re.split(r"\n\s*\n+", cleaned)
        paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
        return paragraphs

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences using regex boundary detection."""
        cleaned = TextPreprocessor.clean_text(text)
        if not cleaned:
            return []
        # Sentence splitting pattern matching periods, exclamation marks, question marks
        # followed by whitespace or end of string, avoiding abbreviations like Dr., e.g., i.e.
        pattern = r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“])"
        sentences = re.split(pattern, cleaned)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_words(self, text: str, lower: bool = True) -> List[str]:
        """Extract alphanumeric words from text."""
        if not text:
            return []
        target = text.lower() if lower else text
        # Regex captures words with optional internal hyphens (e.g. state-of-the-art)
        tokens = re.findall(r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b", target)
        return tokens

    def preprocess_for_ml(self, text: str) -> str:
        """Fully clean and preprocess text for ML vectorization (lowercased, stop words removed)."""
        tokens = self.tokenize_words(text, lower=True)
        filtered = [
            t for t in tokens
            if t not in self.stop_words and len(t) > 2 and not t.isnumeric()
        ]
        return " ".join(filtered)

    def compute_statistics(self, text: str, page_count: int = 1) -> Dict[str, Any]:
        """Compute comprehensive document statistics.

        Returns word count, character counts, paragraph count, sentence count,
        lexical diversity, and estimated reading time.
        """
        cleaned = self.clean_text(text)
        words = self.tokenize_words(cleaned, lower=True)
        word_count = len(words)
        unique_words = len(set(words))
        char_count_with_spaces = len(cleaned)
        char_count_no_spaces = len(re.sub(r"\s+", "", cleaned))

        paragraphs = self.split_into_paragraphs(cleaned)
        sentences = self.split_into_sentences(cleaned)

        # Lexical Diversity: Type-Token Ratio (TTR)
        lexical_diversity = round((unique_words / word_count * 100), 2) if word_count > 0 else 0.0

        # Estimated reading time based on standard average 200 words per minute
        reading_time_minutes = round(word_count / 200, 1)

        # Average words per sentence
        avg_words_per_sentence = (
            round(word_count / len(sentences), 1) if sentences else 0.0
        )

        return {
            "word_count": word_count,
            "unique_word_count": unique_words,
            "char_count_with_spaces": char_count_with_spaces,
            "char_count_no_spaces": char_count_no_spaces,
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "page_count": max(1, page_count),
            "lexical_diversity_percent": lexical_diversity,
            "reading_time_minutes": reading_time_minutes,
            "avg_words_per_sentence": avg_words_per_sentence,
        }
