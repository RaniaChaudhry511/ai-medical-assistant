"""FastAPI entry point for the dual-input medical screening API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.image_model import predict_skin_conditions
from app.text_model import get_symptom_columns, predict_diseases


app = FastAPI(title="AI Medical Screening Tool", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextPredictionRequest(BaseModel):
    symptoms: list[str] = Field(default_factory=list)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the single-page screening interface."""
    return FileResponse(Path(__file__).resolve().parent.parent / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight liveness endpoint that does not require model files."""
    return {"status": "ok"}


@app.get("/symptoms")
def symptoms() -> list[str]:
    """List all valid symptom names for frontend autocomplete."""
    try:
        return get_symptom_columns()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/text")
def predict_text(payload: TextPredictionRequest) -> dict:
    """Return top disease predictions and symptom matching feedback."""
    if not payload.symptoms or not any(symptom.strip() for symptom in payload.symptoms):
        raise HTTPException(status_code=400, detail="Provide at least one symptom.")
    try:
        return predict_diseases(payload.symptoms)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)) -> dict:
    """Return the three most likely skin conditions for an uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    try:
        return predict_skin_conditions(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        await file.close()
