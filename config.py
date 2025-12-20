import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "artifacts", "kyc_logistic_model.pkl")
model_path_scaler = os.path.join(BASE_DIR, "artifacts", "kyc_standard_scaler.pkl")

