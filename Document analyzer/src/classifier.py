"""Document Classifier Module.

Wraps the pre-trained Scikit-learn TF-IDF + Logistic Regression pipeline,
providing real-time inference, class probability distributions, and evaluation metrics.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
import joblib


class DocumentClassifier:
    """Production wrapper for document classification inference and metrics."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "document_classifier.pkl")

        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.classes: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.confusion_matrix: List[List[int]] = []
        self.trained_at: str = ""
        self.train_samples: int = 0
        self.test_samples: int = 0
        self.total_samples: int = 0

        self._load_or_train_model()

    def _load_or_train_model(self) -> None:
        """Load persisted model artifact from disk or train if missing."""
        if not os.path.exists(self.model_path):
            print(f"[INFO] Serialized model not found at {self.model_path}. Initiating training...")
            from models.train_model import train_document_classifier
            train_document_classifier(output_model_path=self.model_path)

        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.vectorizer = bundle["vectorizer"]
        self.classes = bundle.get("classes", list(self.model.classes_))
        self.metrics = bundle.get("metrics", {})
        self.confusion_matrix = bundle.get("confusion_matrix", [])
        self.trained_at = bundle.get("trained_at", "")
        self.train_samples = bundle.get("train_samples", 0)
        self.test_samples = bundle.get("test_samples", 0)
        self.total_samples = bundle.get("total_samples", 0)

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict the category of input document text.

        Returns:
            Dict containing:
                - predicted_category (str)
                - confidence (float): Percentage (0 - 100)
                - confidence_raw (float): Probability (0.0 - 1.0)
                - probabilities (Dict[str, float]): Map of all classes to probability %
                - classes (List[str]): All supported categories
        """
        clean_text = (text or "").strip()
        if not clean_text or len(clean_text) < 10:
            return {
                "predicted_category": "Undetermined",
                "confidence": 0.0,
                "confidence_raw": 0.0,
                "probabilities": {c: 0.0 for c in self.classes},
                "classes": self.classes,
                "message": "Text is too brief for confident classification."
            }

        vec = self.vectorizer.transform([clean_text])
        pred_label = str(self.model.predict(vec)[0])

        # Get calibrated probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            prob_map = {
                cls_name: round(float(prob) * 100, 2)
                for cls_name, prob in zip(self.model.classes_, probs)
            }
            # Sort descending by probability
            sorted_probs = dict(sorted(prob_map.items(), key=lambda item: item[1], reverse=True))
            top_prob = float(probs[list(self.model.classes_).index(pred_label)])
            confidence = round(top_prob * 100, 2)
            raw_conf = float(top_prob)
        else:
            sorted_probs = {c: (100.0 if c == pred_label else 0.0) for c in self.classes}
            confidence = 100.0
            raw_conf = 1.0

        return {
            "predicted_category": pred_label,
            "confidence": confidence,
            "confidence_raw": raw_conf,
            "probabilities": sorted_probs,
            "classes": self.classes,
            "message": "Success"
        }
