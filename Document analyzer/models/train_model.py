"""Machine Learning Model Training and Evaluation Script.

Trains a TF-IDF + Logistic Regression pipeline on the multi-class document dataset.
Evaluates using an 80/20 train/test split, calculates genuine metrics
(Accuracy, Precision, Recall, F1, Confusion Matrix), and persists the artifact.
"""

from __future__ import annotations
from datetime import datetime
import os
import sys
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def get_default_paths() -> tuple[str, str]:
    """Resolve default dataset and model artifact paths."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "data", "sample_dataset.csv")
    model_path = os.path.join(base_dir, "models", "document_classifier.pkl")
    return dataset_path, model_path


def train_document_classifier(
    dataset_path: str | None = None,
    output_model_path: str | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate the Scikit-learn document classification pipeline."""
    default_dataset, default_output = get_default_paths()
    dataset_path = dataset_path or default_dataset
    output_model_path = output_model_path or default_output

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Training dataset not found at: {dataset_path}")

    # Load dataset
    df = pd.read_csv(dataset_path)
    if "category" not in df.columns or "text" not in df.columns:
        raise ValueError("Dataset CSV must contain 'category' and 'text' columns.")

    # Clean text inputs
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10]

    X = [str(t) for t in df["text"]]
    y = [str(c) for c in df["category"]]

    classes = sorted(list(set(y)))
    print(f"[INFO] Loaded {len(df)} samples across {len(classes)} classes: {classes}")

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"[INFO] Dataset split: {len(X_train)} training samples, {len(X_test)} test samples.")

    # TF-IDF Vectorizer with unigrams & bigrams and sublinear TF scaling
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=4000,
        sublinear_tf=True,
        stop_words="english",
        min_df=1,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Calibrated Logistic Regression Classifier
    classifier = LogisticRegression(
        C=4.0,
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )
    classifier.fit(X_train_vec, y_train)

    # Evaluate on held-out test split
    y_pred = classifier.predict(X_test_vec)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    precision_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print("\n" + "=" * 60)
    print("EVALUATION METRICS ON HELD-OUT TEST SPLIT (20%)")
    print("=" * 60)
    print(f"Accuracy:           {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"Macro Precision:    {precision_macro:.4f}")
    print(f"Macro Recall:       {recall_macro:.4f}")
    print(f"Macro F1-Score:     {f1_macro:.4f}")
    print(f"Weighted F1-Score:  {f1_weighted:.4f}")
    print("=" * 60)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Package bundle for deployment
    model_bundle = {
        "model": classifier,
        "vectorizer": vectorizer,
        "classes": classes,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision_macro": round(precision_macro, 4),
            "recall_macro": round(recall_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_weighted": round(precision_weighted, 4),
            "recall_weighted": round(recall_weighted, 4),
            "f1_weighted": round(f1_weighted, 4),
            "report": report_dict,
        },
        "confusion_matrix": cm.tolist(),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": len(df),
        "vocabulary_size": len(vectorizer.vocabulary_),
    }

    # Persist model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(model_bundle, output_model_path)
    print(f"[SUCCESS] Model artifact saved to: {output_model_path}")

    return model_bundle


if __name__ == "__main__":
    train_document_classifier()
