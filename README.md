# AI Medical Assistant

An AI-powered screening tool that lets a user check their health either by describing symptoms in text or by uploading a photo of a skin condition. It combines two separately trained models — a classical Machine Learning model for symptoms and a Deep Learning model for skin images — served through a single FastAPI backend with a simple web UI.

**This is a screening tool, not a diagnostic tool.** It is not a replacement for professional medical advice.

---

## Project Overview

The project has two independent prediction pipelines:

1. **Symptom-based screening (Machine Learning)** — the user selects or types their symptoms, and a trained classifier predicts the most likely disease(s).
2. **Skin image screening (Deep Learning)** — the user uploads a photo of a skin condition, and a trained CNN predicts the most likely skin condition(s).

Both pipelines return their top predictions with confidence scores, and both are exposed through a FastAPI backend with a browser-based UI.

---

## 1. Symptom Classifier (Machine Learning)

- **Model type:** Random Forest Classifier
- **Task:** Multi-class classification across **677 diseases**
- **Input:** A list of symptoms (from a fixed vocabulary of symptom columns)
- **Output:** Top predicted disease(s) with confidence scores
- **Performance:**
  - Top-1 accuracy: **67%**
  - Top-3 accuracy: **76%**
- **Notebook:** `symptom-and-skin-disease-classifier.ipynb`

### Approach
- Symptoms were one-hot encoded against a fixed symptom vocabulary (`symptom_columns.pkl`).
- A Random Forest classifier was trained on the resulting feature matrix to predict disease labels (`disease_labels.pkl`).
- The trained model is stored as `disease_prediction_model.pkl` and hosted on Hugging Face Hub (`RANIA5/ai-medical-assistant`) due to its size — it is downloaded at runtime rather than committed to GitHub.

### Note on prediction quality
Symptoms that are common across many diseases (e.g. fever, cough, body pain) reduce prediction confidence because they don't distinguish between conditions well. More specific, condition-relevant symptoms produce more confident and accurate predictions.

---

## 2. Skin Disease Classifier (Deep Learning)

- **Model type:** Convolutional Neural Network — **MobileNetV2** (transfer learning)
- **Task:** Multi-class classification across **23 skin conditions**
- **Input:** An image of a skin condition (JPG/PNG)
- **Output:** Top predicted condition(s) with confidence scores
- **Performance:** ~21–26% accuracy
- **Notebook:** `skin-disease-image-classifier.ipynb`

### Approach
- Used MobileNetV2 (pretrained on ImageNet) as a feature extractor, fine-tuned on a labeled skin disease image dataset across 23 classes.
- Class labels are stored in `class_indices.json`.
- The trained model is saved as `best_skin_model.h5` and loaded directly by the API at startup.

### Note on prediction quality
Accuracy on the image model is limited — this is an area for future improvement (e.g. more training data, better class balance, deeper fine-tuning, or a different backbone architecture).

---

## 3. Backend API (FastAPI)

The two models are served through a FastAPI application (`app/main.py`), with dedicated modules for each model:
- `app/text_model.py` — loads and runs the symptom classifier
- `app/image_model.py` — loads and runs the skin image classifier

### Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `POST` | `/predict/text` | Accepts a list of symptoms, returns top disease predictions |
| `POST` | `/predict/image` | Accepts an uploaded image file, returns top skin condition predictions |

Interactive API documentation (Swagger UI) is available at `/docs` when the server is running.

### Example request — `/predict/text`
```json
POST /predict/text
{
  "symptoms": ["excessive thirst", "frequent urination", "blurred vision"]
}
```

### Example response
```json
{
  "predictions": [
    { "disease": "Diabetes", "confidence": 82.4 },
    { "disease": "...", "confidence": "..." }
  ]
}
```

---

## 4. Frontend (UI)

A single-page HTML/CSS/JS interface (`index.html`) with two tabs:
- **Check by Symptoms** — click common symptom chips or add custom symptoms, then get predictions.
- **Check by Photo** — drag-and-drop or browse to upload a skin image, then get predictions.

The UI includes a "How to use" guide and a clear disclaimer that results are for screening purposes only.

---

## 5. Deployment & Infrastructure

- **Containerization:** The app is fully Dockerized (`Dockerfile`). The image installs dependencies, downloads the large symptom model from Hugging Face Hub at container startup (`upload_model.py` / runtime download logic), and serves the app via Uvicorn. **Verified working end-to-end in Docker locally.**
- **CI:** A GitHub Actions workflow (`.github/workflows/docker-build.yml`) builds the Docker image on every push to `main`, verifying the build stays healthy.
- **CD / Public deployment:** Public deployment was attempted on three platforms (Hugging Face Spaces, Render, Railway). Hugging Face Spaces and Render required a paid plan for Docker-based hosting on this account; Railway's free tier does not provide enough RAM to load the large symptom model, which causes the `/predict/text` endpoint to crash (502) under the free-tier memory limit. **This is an infrastructure/hosting constraint, not a bug in the model or code** — the same container runs correctly locally and on any host with adequate memory. Public deployment was not a requirement for this project and is planned as a future improvement (alongside model fine-tuning).

---

## Repository Structure

```
├── app/
│   ├── main.py              # FastAPI app and routes
│   ├── text_model.py        # Symptom classifier logic
│   ├── image_model.py       # Skin image classifier logic
│   └── models/
│       ├── best_skin_model.h5
│       ├── class_indices.json
│       ├── disease_labels.pkl
│       └── symptom_columns.pkl
├── index.html                # Frontend UI
├── Dockerfile
├── docker-entrypoint.sh
├── requirements.txt
├── upload_model.py           # Uploads/downloads the large model to/from Hugging Face Hub
├── symptom-and-skin-disease-classifier.ipynb
├── skin-disease-image-classifier.ipynb
└── .github/workflows/docker-build.yml
```

---

## Running Locally

```bash
# Build the image
docker build -t ai-medical-assistant .

# Run the container
docker run -p 8000:8000 ai-medical-assistant
```

Then open `http://localhost:8000` in a browser, or `http://localhost:8000/docs` for the API documentation.

---

## Tech Stack

- **ML:** scikit-learn (Random Forest)
- **DL:** TensorFlow/Keras (MobileNetV2 transfer learning)
- **Backend:** FastAPI, Uvicorn
- **Model hosting:** Hugging Face Hub
- **Containerization:** Docker
- **CI:** GitHub Actions
- **Frontend:** HTML/CSS/JavaScript (vanilla)

---

## Limitations & Future Work

- Symptom model accuracy (67% top-1) leaves room for improvement — better feature engineering or trying gradient-boosted models could help.
- Image model accuracy (21–26%) is the weakest part of the project — more training data, class balancing, and further fine-tuning are the priority next steps.
- The model has not yet been set up to learn from its own prediction errors over time — this is planned for a future version.
- Public deployment is planned once the model is lighter or hosted on infrastructure with sufficient memory.

---

## Disclaimer

This tool provides AI-generated screening insights only. It is **not a substitute for professional medical diagnosis or advice**. Always consult a qualified healthcare provider for any medical concerns.
