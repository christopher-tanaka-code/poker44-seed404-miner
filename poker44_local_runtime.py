from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from build_dataset import chunk_features


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _remap_scores(scores, threshold: float, temperature: float):
    scores = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    z = (scores - float(threshold)) / max(float(temperature), 1e-6)
    return np.clip(_sigmoid(z), 0.0, 1.0)


class LocalWindowsStackPredictor:
    """
    Runtime wrapper for the Windows-trained local stack model.

    Expected artifact keys:
      imputer, base_models, meta_model, calibrator, feature_names, remap, metadata
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Local stack model not found: {self.model_path}")

        self.artifact = joblib.load(self.model_path)

        self.imputer = self.artifact["imputer"]
        self.base_models = self.artifact["base_models"]
        self.meta_model = self.artifact["meta_model"]
        self.calibrator = self.artifact["calibrator"]
        self.feature_names = list(self.artifact["feature_names"])
        self.remap = dict(self.artifact.get("remap") or {})
        self.metadata = dict(self.artifact.get("metadata") or {})

        self.metadata.setdefault("model_name", "poker44_seed404_windows_local_stack")
        self.metadata.setdefault("model_version", "seed404")
        self.metadata.setdefault("framework", "windows-local-stack")
        self.metadata.setdefault(
            "training_data_statement",
            "Public Poker44 benchmark only; miner-visible chunk features only.",
        )
        self.metadata.setdefault(
            "private_data_attestation",
            "No private validator labels or non-public data used.",
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _feature_frame(self, chunks: list[list[dict[str, Any]]]) -> pd.DataFrame:
        rows = []
        for chunk in chunks:
            if not isinstance(chunk, list):
                chunk = []
            row = chunk_features(chunk)
            rows.append(row)

        X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
        X = X.reindex(columns=self.feature_names)
        return X

    def _predict_base_matrix(self, X_values: np.ndarray) -> np.ndarray:
        preds = []

        for _, model in self.base_models.items():
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_values)[:, 1]
            else:
                p = model.predict(X_values)

            preds.append(np.clip(np.asarray(p, dtype=float), 0.0, 1.0))

        return np.vstack(preds).T

    def predict_chunk_scores(self, chunks: list[list[dict[str, Any]]], feature_rows=None) -> list[float]:
        if not chunks:
            return []

        X = self._feature_frame(chunks)
        X_imp = self.imputer.transform(X)

        base_matrix = self._predict_base_matrix(X_imp)
        raw_scores = self.meta_model.predict_proba(base_matrix)[:, 1]

        calibrated = self.calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

        threshold = float(self.remap.get("threshold", 0.5))
        temperature = float(self.remap.get("temperature", 0.25))

        final_scores = _remap_scores(calibrated, threshold=threshold, temperature=temperature)

        return [round(self._clamp01(float(x)), 8) for x in final_scores]
