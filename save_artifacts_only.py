import joblib
import pandas as pd

# load training CSV to recover feature_cols
train_df = pd.read_csv("train_processed.csv")

feature_cols = [
    c for c in train_df.columns
    if c not in ["price", "price_log", "image_path", "id"]
]

# load scaler & spatial
scaler = joblib.load("models/tabular_scaler.pkl")
spatial = joblib.load("models/spatial_center.pkl")

artifacts = {
    "scaler": scaler,
    "feature_cols": feature_cols,
    "center_lat": spatial["center_lat"],
    "center_long": spatial["center_long"]
}

joblib.dump(artifacts, "models/preprocessing_artifacts.pkl")
print("✅ preprocessing_artifacts.pkl saved WITHOUT retraining")
