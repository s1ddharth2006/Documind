# Intelligent Document Analyzer (Documind)

An end-to-end AI/ML document intelligence tool built in Python. It combines text processing, NLP, supervised machine learning classification, and vector-space semantic search — without relying on any external LLM API.

---

## 🌟 Key Features

- **Multi-format ingestion** — extracts text, metadata, and layout from PDF and TXT files via PyMuPDF.
- **Resilient error handling** — handles corrupt PDFs, password protection, scanned image-only documents, and large multi-page files without crashing.
- **Document statistics** — word count, unique vocabulary, character metrics, paragraph/sentence counts, estimated reading time, and lexical diversity (type-token ratio).
- **Text classification** — multi-class classification across 6 categories (Technology, Business, Education, Finance, Sports, News) using a scikit-learn pipeline with real train/test split metrics.
- **TF-IDF keyword extraction** — top unigram/bigram keyphrases with normalized TF-IDF scores.
- **In-document search** — converts a user query into a TF-IDF vector and ranks document sections by cosine similarity, with page-level highlighting.
- **Streamlit dashboard** — KPI cards, probability distribution charts, keyword weight visualizations, and a model evaluation view.
- **Sample documents** — built-in PDF/TXT samples across technology, finance, and sports for quick testing.

---

## 🏗️ System Architecture

```
flowchart TD
    subgraph INGESTION["1. Document Ingestion"]
        A[User Upload / Sample Selector] --> B{Format Check}
        B -->|PDF| C[PyMuPDF Engine]
        B -->|TXT| D[Encoding Normalizer]
        C --> E[Page & Paragraph Segmentation]
        D --> E
    end

    subgraph PREPROCESSING["2. NLP & Preprocessing"]
        E --> F[Text Cleaning & Normalization]
        F --> G[Tokenization & Stop-Word Removal]
        F --> H[Document Statistics Generator]
    end

    subgraph MODELING["3. Machine Learning & TF-IDF"]
        G --> I[TfidfVectorizer unigrams + bigrams]
        I --> J[Calibrated Logistic Regression]
        J --> K[Category & Probability Distribution]
        I --> L[TF-IDF Keyword Extractor]
        L --> M[Top Keyphrases & Weights]
    end

    subgraph RETRIEVAL["4. Vector Search Engine"]
        E --> N[Paragraph Vector Space]
        O[User Query] --> P[Query TF-IDF Vector]
        P --> Q[Cosine Similarity Matcher]
        N --> Q
        Q --> R[Ranked Matches + Snippet Highlighter]
    end

    subgraph DASHBOARD["5. Streamlit Web Interface"]
        H --> S[KPI Metric Cards]
        K --> T[Classification & Probabilities]
        M --> U[Keyword Weights & Badges]
        R --> V[Search Results with Page Attribution]
    end
```

---

## 🔬 How the NLP Pipeline Works

1. **Text normalization** — strips control characters, normalizes smart quotes/dashes, standardizes whitespace and line endings.
2. **Boundary segmentation** — regex-based sentence detection and paragraph splitting, retaining page indices.
3. **Tokenization** — extracts alphanumeric tokens (allowing internal hyphens like `state-of-the-art`), lowercased.
4. **Stop-word removal** — filters common non-informative words while preserving technical terms.
5. **Metrics**:

$$\text{Lexical Diversity (TTR)} = \left(\frac{\text{Unique Word Count}}{\text{Total Word Count}}\right)\times 100\%$$

$$\text{Reading Time} = \frac{\text{Word Count}}{200 \text{ WPM}}$$

---

## 🧠 How the Classifier Works

### Vector Space Representation

Text is converted to TF-IDF vectors using unigrams and bigrams, with sublinear term-frequency scaling (`sublinear_tf=True`) to stop repeated words from dominating.

$$\text{TF}(t,d) = 1+\log(\text{count}(t,d)) \quad \text{IDF}(t,D) = \log\left(\frac{1+|D|}{1+|\{d\in D: t\in d\}|}\right)+1$$

### Classifier

A multi-class **Logistic Regression** model (L2 penalty, C=4.0, L-BFGS solver) separates the 6 categories:

- Technology, Business, Education, Finance, Sports, News

### Evaluation (80/20 stratified split)

| Metric | Score |
|---|---|
| Accuracy | 86.7% |
| Macro Precision | 90.3% |
| Macro Recall | 86.1% |
| Macro F1 | 86.5% |
| Weighted F1 | 86.0% |

Per-class F1: Technology, Sports, Education all at 1.00; News at 0.86; Finance and Business at 0.67 (smaller class sizes and vocabulary overlap between Finance and Business text account for the lower scores here).

---

## 🔎 How Document Search Works

1. Each paragraph/section is extracted with its page number.
2. A section-level TF-IDF matrix is fit across all sections.
3. A user query is transformed into the same vector space.
4. Cosine similarity ranks sections by relevance to the query.
5. Sections above a configurable similarity threshold (default ≥ 0.05) are returned, ranked, with matched terms highlighted.

---

## 📁 Project Structure

```
Document analyzer/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_dataset.csv
├── models/
│   ├── train_model.py
│   └── document_classifier.pkl
├── sample_docs/
│   ├── sample_document.pdf
│   ├── tech_ai_sample.txt
│   ├── finance_report_sample.txt
│   └── sports_championship_sample.txt
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py
│   ├── text_preprocessor.py
│   ├── classifier.py
│   ├── keyword_extractor.py
│   └── document_search.py
└── tests/
    └── test_analyzer.py
```

---

## 🚀 Installation & Local Setup

```bash
git clone https://github.com/s1ddharth2006/Documind.git
cd "Document analyzer"
python -m pip install -r requirements.txt
```

### Retrain the model (optional)

```bash
python models/train_model.py
```

### Run tests

```bash
python -m pytest tests/test_analyzer.py -v
```

Expected: `13 passed`.

### Launch the app

```bash
python -m streamlit run app.py
```

Then open `http://localhost:8501`.

---

## 💡 Example Walkthrough

1. Open the app, go to **⚡ Quick Test** in the sidebar, select `AI & Cloud Architecture (PDF)`.
2. Review the KPI cards — page count, word count, lexical diversity, reading time.
3. Check the predicted category (`Technology`) and confidence distribution.
4. Inspect the TF-IDF keyword list (e.g. `neural`, `vector`, `processing`).
5. Search the document for something like `cosine similarity` and see the ranked, highlighted matches with page numbers.

---

## 🔮 Future Enhancements

- Named entity recognition (organizations, people, locations) via spaCy.
- Extractive summarization using TextRank.
- Export document statistics and classification results as PDF/CSV reports.

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
