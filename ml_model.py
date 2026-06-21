"""
ml_model.py - Scikit-learn Random Forest classifier for road damage priority.

Usage:
    # Train and save model
    python ml_model.py

    # In application code
    model = DamagePriorityModel()
    model.load()
    result = model.predict({...features...})
"""

import os
import csv
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

MODEL_PATH = os.path.join("models", "damage_classifier.joblib")
ENCODER_PATH = os.path.join("models", "label_encoder.joblib")
DATASET_FILE = "road_damage_dataset.csv"

FEATURE_COLS = [
    "damage_percentage", "num_damaged_regions", "crack_density",
    "texture_roughness", "dark_surface_percentage"
]

PRIORITY_ORDER = ["Low Priority", "Medium Priority", "High Priority", "Critical Priority"]


class DamagePriorityModel:
    """Wraps a trained Random Forest classifier for road damage prioritization."""

    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self.encoder: LabelEncoder | None = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, dataset_path: str = DATASET_FILE):
        """Load dataset, train the RF model, print metrics, save artifacts."""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at '{dataset_path}'. "
                "Run generate_dataset.py first."
            )

        X, y_raw = self._load_csv(dataset_path)

        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(y_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(
            y_test, y_pred, target_names=self.encoder.classes_
        ))

        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.encoder, ENCODER_PATH)
        print(f"Model saved → {MODEL_PATH}")
        self._loaded = True

    def _load_csv(self, path: str):
        X, y = [], []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                X.append([float(row[col]) for col in FEATURE_COLS])
                y.append(row["priority"])
        return np.array(X), np.array(y)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def load(self):
        """Load pre-trained model artifacts from disk."""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(
                "Model files not found. Run 'python ml_model.py' to train the model."
            )
        self.model = joblib.load(MODEL_PATH)
        self.encoder = joblib.load(ENCODER_PATH)
        self._loaded = True

    def predict(self, features: dict) -> dict:
        """
        Predict priority and return confidence scores.

        Args:
            features: dict with keys matching FEATURE_COLS

        Returns:
            dict with 'priority' (str), 'confidence' (float), 'probabilities' (dict)
        """
        if not self._loaded:
            self.load()

        X = np.array([[features[col] for col in FEATURE_COLS]])
        label_idx = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        classes = self.encoder.classes_
        priority = self.encoder.inverse_transform([label_idx])[0]
        confidence = round(float(proba[label_idx]) * 100, 1)
        probabilities = {
            cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)
        }

        return {
            "priority": priority,
            "confidence": confidence,
            "probabilities": probabilities,
        }


# ------------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------------
if __name__ == "__main__":
    import subprocess, sys

    if not os.path.exists(DATASET_FILE):
        print("Dataset not found — generating it first...")
        subprocess.run([sys.executable, "generate_dataset.py"], check=True)

    model = DamagePriorityModel()
    model.train()