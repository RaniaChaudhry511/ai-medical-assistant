from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="app/models/disease_prediction_model.pkl",
    path_in_repo="disease_prediction_model.pkl",
    repo_id="RANIA5/ai-medical-assistant",
    repo_type="model"
)
print("Upload complete!")
