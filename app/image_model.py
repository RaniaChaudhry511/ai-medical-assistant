"""Skin-image classification helpers."""

from __future__ import annotations

from functools import lru_cache
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


MODEL_DIR = Path(__file__).resolve().parent / "models"


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[Any, dict[str, str]]:
    """Load the Keras classifier and output-class mapping once."""
    try:
        # Imported here so the text endpoints remain usable if TensorFlow is
        # unavailable during an initial deployment.
        from tensorflow.keras.models import load_model

        model = load_model(MODEL_DIR / "best_skin_model.h5")
        with (MODEL_DIR / "class_indices.json").open(encoding="utf-8") as file:
            class_indices = json.load(file)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Image model files are missing. Place the .h5 and .json files in app/models/."
        ) from exc
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Unable to load image model artefacts.") from exc

    if not isinstance(class_indices, dict):
        raise RuntimeError("class_indices.json must contain an index-to-name object.")
    return model, {str(index): str(name) for index, name in class_indices.items()}


def predict_skin_conditions(image_bytes: bytes) -> dict[str, list[dict[str, float | str]]]:
    """Classify image bytes and return the top three skin-condition results."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB").resize((224, 224))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    model, class_indices = _load_artifacts()
    batch = np.expand_dims(np.asarray(image, dtype=np.float32) / 255.0, axis=0)
    probabilities = np.asarray(model.predict(batch, verbose=0))[0]
    top_indices = np.argsort(probabilities)[::-1][: min(3, len(probabilities))]
    return {
        "predictions": [
            {
                "condition": class_indices.get(str(int(index)), f"Class {int(index)}"),
                "confidence": round(float(probabilities[index]) * 100, 2),
            }
            for index in top_indices
        ]
    }
