#!/bin/sh
set -eu

MODEL_PATH="app/models/disease_prediction_model.pkl"
MODEL_URL="https://huggingface.co/RANIA5/ai-medical-assistant/resolve/main/disease_prediction_model.pkl"

if [ ! -f "$MODEL_PATH" ]; then
    mkdir -p "$(dirname "$MODEL_PATH")"
    curl -fsSL --retry 3 "$MODEL_URL" -o "${MODEL_PATH}.tmp"
    mv "${MODEL_PATH}.tmp" "$MODEL_PATH"
fi

exec "$@"
