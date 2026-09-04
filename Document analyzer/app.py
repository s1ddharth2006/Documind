"""Intelligent Document Analyzer - Streamlit Application.

A portfolio-grade web application demonstrating Python, PyMuPDF, Scikit-learn,
NLP preprocessing, TF-IDF vectorization, document classification, and vector search.
"""

from __future__ import annotations
import os
import io
from typing import Any, Dict, List, Optional
import altair as alt
import pandas as pd
import streamlit as st

from src.classifier import DocumentClassifier
from src.document_search import DocumentSearchEngine
from src.keyword_extractor import TfidfKeywordExtractor
from src.pdf_processor import DocumentProcessingError, DocumentProcessor
from src.text_preprocessor import TextPreprocessor


# ---------------------------------------------------------------------------
# Streamlit Page Configuration & Modern Theme Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern visual polish
CUSTOM_CSS = """
<style>
    /* Global Typography & Font Setup */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* App Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        color: #f8fafc;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        line-height: 1.6;
        max-width: 850px;
        margin-bottom: 1.2rem;
    }
    .badge-tag {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        margin-right: 0.5rem;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #cbd5e1;
    }

    /* KPI Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: #1e293b;
            border-color: #334155;
            color: #f8fafc;
        }
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    @media (prefers-color-scheme: dark) {
        .metric-value {
            color: #f1f5f9;
        }
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }

    /* Classification Hero Result Card */
    .classification-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    @media (prefers-color-scheme: dark) {
        .classification-box {
            background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
            border-color: #059669;
        }
    }
    .classification-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #15803d;
        margin-bottom: 0.5rem;
    }
    @media (prefers-color-scheme: dark) {
        .classification-title {
            color: #86efac;
        }
    }
    .classification-label {
        font-size: 2rem;
        font-weight: 800;
        color: #14532d;
    }
    @media (prefers-color-scheme: dark) {
        .classification-label {
            color: #f0fdf4;
        }
    }

    /* Search Result Section */
    .search-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    @media (prefers-color-scheme: dark) {
        .search-card {
            background: #1e293b;
            border-color: #334155;
            border-left-color: #60a5fa;
            color: #e2e8f0;
        }
    }
    .search-meta {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 0.6rem;
    }
    .search-score-badge {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    @media (prefers-color-scheme: dark) {
        .search-score-badge {
            background: #1e3a8a;
            color: #bfdbfe;
        }
    }
    .search-snippet {
        font-size: 0.95rem;
        line-height: 1.65;
        color: #334155;
    }
    @media (prefers-color-scheme: dark) {
        .search-snippet {
            color: #cbd5e1;
        }
    }
    mark.search-match {
        background-color: #fef08a !important;
        color: #854d0e !important;
        padding: 0.15rem 0.35rem;
        border-radius: 4px;
        font-weight: 600;
    }

    /* Keyword Pill */
    .kw-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.8rem;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #334155;
        margin: 0.25rem;
    }
    @media (prefers-color-scheme: dark) {
        .kw-pill {
            background: #334155;
            border-color: #475569;
            color: #f1f5f9;
        }
    }
    .kw-weight {
        font-size: 0.75rem;
        font-weight: 700;
        color: #2563eb;
        background: #dbeafe;
        padding: 0.1rem 0.4rem;
        border-radius: 9999px;
    }
    @media (prefers-color-scheme: dark) {
        .kw-weight {
            background: #1e3a8a;
            color: #93c5fd;
        }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Machine Learning Classifier...")
def get_classifier() -> DocumentClassifier:
    """Load cached Scikit-learn document classifier."""
    return DocumentClassifier()


@st.cache_resource
def get_keyword_extractor() -> TfidfKeywordExtractor:
    """Initialize TF-IDF keyword extractor."""
    return TfidfKeywordExtractor(top_n=20)


@st.cache_resource
def get_preprocessor() -> TextPreprocessor:
    """Initialize text preprocessor and statistics generator."""
    return TextPreprocessor()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def load_sample_file(sample_name: str) -> tuple[bytes, str]:
    """Load built-in sample document for instant testing."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_map = {
        "AI & Cloud Architecture (PDF)": ("sample_docs/sample_document.pdf", "sample_document.pdf"),
        "Artificial Intelligence & ML (TXT)": ("sample_docs/tech_ai_sample.txt", "tech_ai_sample.txt"),
        "Capital Markets & Macroeconomics (TXT)": ("sample_docs/finance_report_sample.txt", "finance_report_sample.txt"),
        "Sports Science & Tactics (TXT)": ("sample_docs/sports_championship_sample.txt", "sports_championship_sample.txt"),
    }
    rel_path, filename = file_map[sample_name]
    abs_path = os.path.join(base_dir, rel_path)
    with open(abs_path, "rb") as f:
        content = f.read()
    return content, filename


# ---------------------------------------------------------------------------
# Sidebar: Navigation, Sample Files & Model Inspector
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/document.png", width=64)
    st.title("Document Analyzer")
    st.caption("AI/ML Portfolio Project • End-to-End NLP")

    st.markdown("---")
    st.subheader("⚡ Quick Test")
    st.write("Test instantly without uploading:")
    sample_options = [
        "Select a sample document...",
        "AI & Cloud Architecture (PDF)",
        "Artificial Intelligence & ML (TXT)",
        "Capital Markets & Macroeconomics (TXT)",
        "Sports Science & Tactics (TXT)",
    ]
    selected_sample = st.selectbox(
        "Built-in Sample Documents",
        sample_options,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.subheader("⚙️ Search Configuration")
    search_top_k = st.slider("Max Search Results", min_value=1, max_value=10, value=4)
    search_min_score = st.slider(
        "Min Similarity Threshold",
        min_value=0.0,
        max_value=0.5,
        value=0.05,
        step=0.01,
        help="Filters out matches with cosine similarity below this score.",
    )

    st.markdown("---")
    st.subheader("🧠 Model Performance")
    clf = get_classifier()
    metrics = clf.metrics

    if metrics:
        st.write(f"**Trained On:** {clf.total_samples} samples across 6 classes")
        st.write(f"**Test Accuracy:** `{metrics.get('accuracy', 0.0) * 100:.1f}%`")
        st.write(f"**Macro F1-Score:** `{metrics.get('f1_macro', 0.0):.4f}`")
        st.write(f"**Macro Precision:** `{metrics.get('precision_macro', 0.0):.4f}`")
        st.write(f"**Macro Recall:** `{metrics.get('recall_macro', 0.0):.4f}`")

        with st.expander("📊 View Confusion Matrix & Classes"):
            st.caption("Confusion Matrix on 20% Held-Out Test Split:")
            cm_df = pd.DataFrame(
                clf.confusion_matrix,
                index=[f"True: {c}" for c in clf.classes],
                columns=[f"Pred: {c}" for c in clf.classes],
            )
            st.dataframe(cm_df, use_container_width=True)

            st.caption("Class Breakdown:")
            st.write(", ".join([f"`{c}`" for c in clf.classes]))

    st.markdown("---")
    st.caption("Built with Python, Streamlit, PyMuPDF & Scikit-learn.")


# ---------------------------------------------------------------------------
# Main Header / Hero Banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">Intelligent Document Analyzer</div>
        <div class="hero-subtitle">
            An end-to-end NLP and machine learning platform for automated document ingestion,
            structural statistical analysis, supervised multi-class categorization, TF-IDF keyword extraction,
            and cosine similarity vector retrieval.
        </div>
        <div>
            <span class="badge-tag">PyMuPDF Ingestion</span>
            <span class="badge-tag">TF-IDF Vector Space</span>
            <span class="badge-tag">Scikit-learn Logistic Regression</span>
            <span class="badge-tag">Cosine Similarity Search</span>
            <span class="badge-tag">Deterministic NLP</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Upload / Input Section
# ---------------------------------------------------------------------------
st.subheader("📂 Document Ingestion")

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT document for automated NLP analysis",
    type=["pdf", "txt"],
    help="Accepts .pdf and .txt documents up to 50MB.",
)

active_bytes: Optional[bytes] = None
active_filename: str = ""

if uploaded_file is not None:
    active_bytes = uploaded_file.getvalue()
    active_filename = uploaded_file.name
elif selected_sample != "Select a sample document...":
    active_bytes, active_filename = load_sample_file(selected_sample)
    st.info(f"Loaded built-in sample document: **{active_filename}**")

if not active_bytes:
    # Landing instructions when no file is active
    st.markdown(
        """
        <div style="background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 3rem 2rem; text-align: center; margin-top: 1rem;">
            <h3 style="color: #475569; margin-bottom: 0.5rem;">No Document Loaded</h3>
            <p style="color: #64748b; max-width: 600px; margin: 0 auto 1.5rem auto;">
                Drag and drop a <strong>PDF</strong> or <strong>TXT</strong> file above, or select one of the
                instant sample documents from the left sidebar to analyze text statistics, machine learning classification,
                and vector search.
            </p>
            <div style="display: flex; justify-content: center; gap: 1rem;">
                <span style="background: #e2e8f0; color: #334155; padding: 0.4rem 0.8rem; border-radius: 6px; font-weight: 600; font-size: 0.85rem;">
                    📄 PDF Documents
                </span>
                <span style="background: #e2e8f0; color: #334155; padding: 0.4rem 0.8rem; border-radius: 6px; font-weight: 600; font-size: 0.85rem;">
                    📝 Plain Text (.txt)
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ---------------------------------------------------------------------------
# Document Processing & Ingestion Pipeline
# ---------------------------------------------------------------------------
with st.spinner("Processing document and extracting text..."):
    try:
        doc_result = DocumentProcessor.process_file(active_bytes, filename=active_filename)
    except DocumentProcessingError as e:
        st.error(f"❌ Document Ingestion Error: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error while processing document: {str(e)}")
        st.stop()

# Display any non-fatal document warnings
if doc_result.get("warnings"):
    for warn in doc_result["warnings"]:
        st.warning(f"⚠️ {warn}")

raw_text = doc_result["raw_text"]
if not raw_text.strip():
    st.error("❌ No text could be extracted from this document. Please ensure the document is not an empty or scanned image file.")
    st.stop()

# Precompute stats, classification, keywords, and search engine
preprocessor = get_preprocessor()
stats = preprocessor.compute_statistics(raw_text, page_count=doc_result["page_count"])

classifier = get_classifier()
classification_result = classifier.predict(raw_text)

keyword_extractor = get_keyword_extractor()
keywords = keyword_extractor.extract_keywords(raw_text, top_n=15)

search_engine = DocumentSearchEngine(min_similarity_threshold=search_min_score)
indexed_sections_count = search_engine.index_document(raw_text, pages=doc_result["pages"])


# ---------------------------------------------------------------------------
# Section 1: Basic Document Statistics
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Document Overview & Statistics")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Pages</div>
            <div class="metric-value">{stats['page_count']}</div>
            <div class="metric-sub">{doc_result['file_type'].upper()} Format</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Words</div>
            <div class="metric-value">{stats['word_count']:,}</div>
            <div class="metric-sub">{stats['unique_word_count']:,} unique words</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Characters</div>
            <div class="metric-value">{stats['char_count_with_spaces']:,}</div>
            <div class="metric-sub">{stats['char_count_no_spaces']:,} non-space</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Paragraphs</div>
            <div class="metric-value">{stats['paragraph_count']}</div>
            <div class="metric-sub">{stats['sentence_count']} sentences</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Lexical Diversity</div>
            <div class="metric-value">{stats['lexical_diversity_percent']}%</div>
            <div class="metric-sub">Type-Token Ratio</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col6:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Reading Time</div>
            <div class="metric-value">{stats['reading_time_minutes']} min</div>
            <div class="metric-sub">~200 WPM pace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section 2: Machine Learning Classification & Top Keywords (2 Columns)
# ---------------------------------------------------------------------------
st.markdown("---")
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("🏷️ Machine Learning Document Classification")

    top_cat = classification_result["predicted_category"]
    conf = classification_result["confidence"]

    category_icons = {
        "Technology": "💻",
        "Business": "💼",
        "Education": "🎓",
        "Finance": "📈",
        "Sports": "⚽",
        "News": "📰",
    }
    icon = category_icons.get(top_cat, "📄")

    st.markdown(
        f"""
        <div class="classification-box">
            <div class="classification-title">Predicted Category • Scikit-learn Classifier</div>
            <div class="classification-label">{icon} {top_cat}</div>
            <div style="font-size: 1rem; color: #166534; margin-top: 0.4rem; font-weight: 600;">
                Model Confidence: <span style="font-size: 1.2rem;">{conf}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Class Probability Distribution:")
    prob_df = pd.DataFrame(
        list(classification_result["probabilities"].items()),
        columns=["Category", "Probability (%)"],
    )

    # Altair horizontal bar chart for probability distribution
    chart_prob = (
        alt.Chart(prob_df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("Probability (%):Q", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Category:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.Category == top_cat,
                alt.value("#22c55e"),  # Highlight top class in vibrant green
                alt.value("#94a3b8"),  # Other classes in muted slate
            ),
            tooltip=["Category", "Probability (%)"],
        )
        .properties(height=220)
    )
    st.altair_chart(chart_prob, use_container_width=True)


with col_right:
    st.subheader("🔑 Salient Keywords (TF-IDF)")

    if keywords:
        # Altair bar chart for top keywords
        kw_df = pd.DataFrame(keywords[:10])
        kw_chart = (
            alt.Chart(kw_df)
            .mark_bar(color="#6366f1", cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X("score:Q", title="Normalized TF-IDF Score", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("keyword:N", sort="-x", title=None),
                tooltip=["keyword", "score", "frequency", "ngram_type"],
            )
            .properties(height=220)
        )
        st.altair_chart(kw_chart, use_container_width=True)

        st.caption("Top 15 Extracted Keywords & Keyphrases:")
        pills_html = '<div style="display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem;">'
        for kw in keywords:
            pills_html += (
                f'<span class="kw-pill">'
                f'<span>{kw["keyword"]}</span>'
                f'<span class="kw-weight">{kw["score"]}</span>'
                f'</span>'
            )
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)
    else:
        st.info("No distinctive keywords could be extracted from this text.")


# ---------------------------------------------------------------------------
# Section 3: Semantic-ish Document Search (TF-IDF + Cosine Similarity)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 In-Document Search (TF-IDF + Cosine Similarity)")
st.caption(
    "Query paragraphs using vector space cosine similarity. "
    f"Indexed **{indexed_sections_count}** sections across **{stats['page_count']}** page(s)."
)

search_query = st.text_input(
    "Enter a search query or topic to locate relevant sections:",
    placeholder="e.g. neural network accelerators, inflation policy, periodization training...",
)

if search_query.strip():
    search_results = search_engine.search(
        search_query,
        top_k=search_top_k,
        threshold=search_min_score,
    )

    if search_results:
        st.write(f"Found **{len(search_results)}** relevant section(s) above similarity threshold `{search_min_score}`:")
        for res in search_results:
            matched_terms_str = ", ".join([f"`{t}`" for t in res["matched_terms"]]) if res["matched_terms"] else "None"
            st.markdown(
                f"""
                <div class="search-card">
                    <div class="search-meta">
                        <span>PAGE {res['page_number']} • SECTION #{res['section_id']}</span>
                        <span class="search-score-badge">Cosine Similarity: {res['score_percent']}%</span>
                        <span style="margin-left: 0.75rem; color: #64748b; font-size: 0.75rem;">Matched Terms: {matched_terms_str}</span>
                    </div>
                    <div class="search-snippet">
                        {res['highlighted_text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning(
            f"No matching sections found with similarity ≥ {search_min_score}. "
            "Try lowering the threshold in the sidebar or using different query keywords."
        )
else:
    st.info("💡 Type a search query above to locate relevant sections with highlighted terms and relevance scores.")


# ---------------------------------------------------------------------------
# Section 4: Document Preview & Preprocessing Inspector
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📖 Document Content & Processing Inspector")

tab1, tab2, tab3 = st.tabs(["📄 Formatted Document Viewer", "🧹 Preprocessed Clean Tokens", "📑 Page-by-Page Breakdown"])

with tab1:
    st.caption(f"Full Extracted Text ({doc_result['file_name']}):")
    st.text_area(
        "Extracted Document Text",
        raw_text,
        height=320,
        label_visibility="collapsed",
    )

with tab2:
    st.caption("Normalized and tokenized stream used by the Scikit-learn ML pipeline:")
    clean_stream = preprocessor.preprocess_for_ml(raw_text)
    st.text_area(
        "Preprocessed Tokens (Lowercased, Stop-Words Removed)",
        clean_stream,
        height=220,
        label_visibility="collapsed",
    )

with tab3:
    st.caption(f"Document contains {len(doc_result['pages'])} page(s):")
    for page in doc_result["pages"]:
        with st.expander(f"Page {page['page_number']} ({page.get('char_count', len(page['text']))} characters)"):
            st.text(page["text"])


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1.5rem 0;">
        <strong>Intelligent Document Analyzer</strong> • AI Developer Internship Portfolio Project<br>
        Engineered with Python, Streamlit, PyMuPDF, Scikit-learn, and Cosine Similarity Vector Space.
    </div>
    """,
    unsafe_allow_html=True,
)
