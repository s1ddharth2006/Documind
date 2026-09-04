"""Unit and Integration Tests for Intelligent Document Analyzer.

Tests all core components:
1. Document Ingestion (PDF & TXT, error cases, empty files)
2. Text Preprocessor (tokenization, cleaning, statistics)
3. Document Classifier (inference, confidence, probability distribution, metrics)
4. TF-IDF Keyword Extractor (weights, frequencies, n-grams)
5. Document Search Engine (indexing, vector similarity, term highlighting)
"""

import os
import pytest
from src.classifier import DocumentClassifier
from src.document_search import DocumentSearchEngine
from src.keyword_extractor import TfidfKeywordExtractor
from src.pdf_processor import DocumentProcessingError, DocumentProcessor
from src.text_preprocessor import TextPreprocessor


@pytest.fixture
def sample_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_docs")


@pytest.fixture
def tech_sample_text(sample_dir):
    path = os.path.join(sample_dir, "tech_ai_sample.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def finance_sample_text(sample_dir):
    path = os.path.join(sample_dir, "finance_report_sample.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def sports_sample_text(sample_dir):
    path = os.path.join(sample_dir, "sports_championship_sample.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 1. Document Processor Tests
# ---------------------------------------------------------------------------
def test_pdf_extraction_valid(sample_dir):
    pdf_path = os.path.join(sample_dir, "sample_document.pdf")
    assert os.path.exists(pdf_path), "Sample PDF must exist"

    result = DocumentProcessor.process_file(pdf_path, filename="sample_document.pdf")
    assert result["file_type"] == "pdf"
    assert result["page_count"] == 3
    assert len(result["pages"]) == 3
    assert "Executive Technical Report" in result["raw_text"]
    assert len(result["raw_text"]) > 200
    assert result["pages"][0]["page_number"] == 1


def test_txt_extraction_valid(sample_dir):
    txt_path = os.path.join(sample_dir, "tech_ai_sample.txt")
    assert os.path.exists(txt_path), "Sample TXT must exist"

    result = DocumentProcessor.process_file(txt_path, filename="tech_ai_sample.txt")
    assert result["file_type"] == "txt"
    assert result["page_count"] == 1
    assert "artificial intelligence" in result["raw_text"].lower()


def test_unsupported_file_format():
    with pytest.raises(DocumentProcessingError, match="Unsupported file format"):
        DocumentProcessor.process_file(b"some binary content", filename="report.docx")


def test_empty_file_error():
    with pytest.raises(DocumentProcessingError, match="empty"):
        DocumentProcessor.process_file(b"", filename="empty.txt")


# ---------------------------------------------------------------------------
# 2. Text Preprocessor Tests
# ---------------------------------------------------------------------------
def test_text_cleaning_and_normalization():
    raw = "  Hello   World!\r\n\r\n\r\nThis is “smart quotes” and an — em-dash.  \t  "
    cleaned = TextPreprocessor.clean_text(raw)
    assert "“" not in cleaned and "”" not in cleaned
    assert '"smart quotes"' in cleaned
    assert "em-dash" in cleaned
    assert "\r" not in cleaned


def test_paragraph_and_sentence_segmentation():
    text = "First paragraph sentence one. Sentence two!\n\nSecond paragraph here? Yes."
    paragraphs = TextPreprocessor.split_into_paragraphs(text)
    assert len(paragraphs) == 2

    sentences = TextPreprocessor.split_into_sentences(text)
    assert len(sentences) >= 3


def test_compute_statistics(tech_sample_text):
    preprocessor = TextPreprocessor()
    stats = preprocessor.compute_statistics(tech_sample_text, page_count=1)

    assert stats["word_count"] > 100
    assert stats["char_count_with_spaces"] > stats["char_count_no_spaces"]
    assert stats["paragraph_count"] >= 4
    assert stats["sentence_count"] >= 4
    assert 0 < stats["lexical_diversity_percent"] <= 100
    assert stats["reading_time_minutes"] > 0


# ---------------------------------------------------------------------------
# 3. Document Classifier Tests
# ---------------------------------------------------------------------------
def test_classifier_inference(tech_sample_text, sports_sample_text):
    clf = DocumentClassifier()
    assert len(clf.classes) == 6
    assert "Technology" in clf.classes
    assert "Sports" in clf.classes

    # Test Tech Prediction
    tech_pred = clf.predict(tech_sample_text)
    assert tech_pred["predicted_category"] == "Technology"
    assert tech_pred["confidence"] > 40.0  # More than double random baseline (16.7%)
    assert len(tech_pred["probabilities"]) == 6

    # Test Sports Prediction
    sports_pred = clf.predict(sports_sample_text)
    assert sports_pred["predicted_category"] == "Sports"
    assert sports_pred["confidence"] > 40.0

    # Test Empty Text Edge Case
    empty_pred = clf.predict("")
    assert empty_pred["predicted_category"] == "Undetermined"
    assert empty_pred["confidence"] == 0.0


def test_classifier_metrics():
    clf = DocumentClassifier()
    metrics = clf.metrics
    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0.70
    assert "precision_macro" in metrics
    assert "f1_macro" in metrics
    assert len(clf.confusion_matrix) == 6


# ---------------------------------------------------------------------------
# 4. Keyword Extractor Tests
# ---------------------------------------------------------------------------
def test_keyword_extraction(tech_sample_text):
    extractor = TfidfKeywordExtractor(top_n=10)
    keywords = extractor.extract_keywords(tech_sample_text)

    assert len(keywords) == 10
    first = keywords[0]
    assert "keyword" in first
    assert "score" in first
    assert "frequency" in first
    assert first["score"] == 1.0  # Normalized top score

    kw_names = [k["keyword"].lower() for k in keywords]
    assert any("intelligence" in k or "neural" in k or "learning" in k or "vector" in k for k in kw_names)


def test_keyword_extraction_empty():
    extractor = TfidfKeywordExtractor()
    assert extractor.extract_keywords("") == []
    assert extractor.extract_keywords("a the is") == []


# ---------------------------------------------------------------------------
# 5. Document Search Engine Tests
# ---------------------------------------------------------------------------
def test_document_search(tech_sample_text):
    search_engine = DocumentSearchEngine(min_similarity_threshold=0.01)
    num_sections = search_engine.index_document(tech_sample_text)
    assert num_sections >= 4

    # Search for concept in section 1
    results = search_engine.search("GPU clusters neural networks parallel", top_k=3)
    assert len(results) > 0
    top_result = results[0]
    assert top_result["score"] > 0.1
    assert "search-match" in top_result["highlighted_text"]
    assert any("gpu" in t.lower() or "neural" in t.lower() for t in top_result["matched_terms"])

    # Search for concept in section 3
    edge_results = search_engine.search("edge computing TinyML devices", top_k=2)
    assert len(edge_results) > 0
    assert any("edge" in r["raw_text"].lower() for r in edge_results)


def test_search_empty_query(tech_sample_text):
    search_engine = DocumentSearchEngine()
    search_engine.index_document(tech_sample_text)
    assert search_engine.search("") == []
    assert search_engine.search("   ") == []
