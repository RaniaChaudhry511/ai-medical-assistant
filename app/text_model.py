"""Symptom-based disease prediction helpers.

The model artefacts are intentionally loaded lazily.  This keeps the API's
health endpoint usable while model files are being copied into ``app/models``.
"""

from __future__ import annotations

import difflib
from functools import lru_cache
from pathlib import Path
import pickle
import joblib
from typing import Any

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent / "models"


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[Any, list[str], list[str]]:
    """Load the Random Forest and its feature/label metadata once."""
    try:
        with (MODEL_DIR / "disease_prediction_model.pkl").open("rb") as file:
            model = joblib.load(file)
        with (MODEL_DIR / "symptom_columns.pkl").open("rb") as file:
            symptom_columns = joblib.load(file)
        with (MODEL_DIR / "disease_labels.pkl").open("rb") as file:
            disease_labels = joblib.load(file)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Text model files are missing. Place the .pkl files in app/models/."
        ) from exc
    except (OSError, pickle.UnpicklingError) as exc:
        raise RuntimeError("Unable to load text model artefacts.") from exc

    if not isinstance(symptom_columns, list) or not isinstance(disease_labels, list):
        raise RuntimeError("Symptom columns and disease labels must be Python lists.")
    return model, symptom_columns, disease_labels


def get_symptom_columns() -> list[str]:
    """Return a copy of the valid symptoms for API consumers."""
    _, symptom_columns, _ = _load_artifacts()
    return list(symptom_columns)


def _match_symptoms(
    user_symptoms: list[str], symptom_columns: list[str]
) -> tuple[list[str], list[str]]:
    """Match input case-insensitively: exact names first, then substring names."""
    normalized = {name.casefold(): name for name in symptom_columns}
    matched: list[str] = []
    unmatched: list[str] = []

    for raw_symptom in user_symptoms:
        if not isinstance(raw_symptom, str) or not raw_symptom.strip():
            unmatched.append(str(raw_symptom))
            continue

        query = raw_symptom.strip().casefold()
        match = normalized.get(query)
        if match is None:
            # A short user term ("head") can match "headache"; accepting the
            # reverse direction also handles descriptive input such as
            # "severe headache" for a canonical "headache" feature.
            match = next(
                (column for column in symptom_columns
                 if query in column.casefold() or column.casefold() in query),
                None,
            )
        if match is None:
            close_matches = difflib.get_close_matches(
                query, normalized.keys(), n=1, cutoff=0.75
            )
            if close_matches:
                match = normalized[close_matches[0]]

        if match is None:
            unmatched.append(raw_symptom)
        elif match not in matched:
            matched.append(match)

    return matched, unmatched


def _disease_name(model_class: Any, disease_labels: list[str]) -> str:
    """Map a classifier class value to its display label safely."""
    if isinstance(model_class, (int, np.integer)) and 0 <= int(model_class) < len(disease_labels):
        return str(disease_labels[int(model_class)])
    return str(model_class)


def predict_diseases(user_symptoms: list[str]) -> dict[str, list[dict[str, float | str]] | list[str]]:
    """Predict the three most likely diseases from typed symptom names."""
    model, symptom_columns, disease_labels = _load_artifacts()
    matched, unmatched = _match_symptoms(user_symptoms, symptom_columns)

    feature_index = {name: index for index, name in enumerate(symptom_columns)}
    features = np.zeros((1, len(symptom_columns)), dtype=np.int8)
    for symptom in matched:
        features[0, feature_index[symptom]] = 1

    probabilities = np.asarray(model.predict_proba(features))[0]
    classes = np.asarray(model.classes_)
    top_indices = np.argsort(probabilities)[::-1][: min(3, len(probabilities))]
    top_probability_sum = float(probabilities[top_indices].sum())
    predictions = [
        {
            "disease": _disease_name(classes[index], disease_labels),
            "confidence": round(float(probabilities[index]) * 100, 2),
            "relative_confidence": round(
                (float(probabilities[index]) / top_probability_sum) * 100, 2)
                if top_probability_sum
                else 0.0,
        }
        for index in top_indices
    ]
    return {
        "predictions": predictions,
        "matched_symptoms": matched,
        "unmatched_symptoms": unmatched,
    }
