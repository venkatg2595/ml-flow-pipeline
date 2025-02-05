import os
import numpy as np
import joblib
import argparse
from sklearn.ensemble import RandomForestClassifier
from google.cloud import storage

# ----------------------------
# ✅ Train and Save Model
# ----------------------------
def train_and_save_model(X_train, y_train, model_output_path):
    print("🚀 Training RandomForest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Ensure output directory exists
    os.makedirs(model_output_path, exist_ok=True)

    # Save the trained model
    model_filename = os.path.join(model_output_path, 'model.joblib')
    joblib.dump(model, model_filename)
    print(f"✅ Model saved locally: {model_filename}")

    # Upload model to Google Cloud Storage (GCS)
    upload_to_gcs(model_filename, "mlops-bucket", "models/model.joblib")

# ----------------------------
# ✅ Upload Model to GCS
# ----------------------------
def upload_to_gcs(local_file, bucket_name, gcs_path):
    print("📡 Uploading model to Google Cloud Storage...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_file)
    print(f"✅ Model uploaded to GCS: gs://{bucket_name}/{gcs_path}")

# ----------------------------
# ✅ Main Execution
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-output-path', dest='model_output_path', default='model', type=str)
    args = parser.parse_args()

    # Load processed training data
    print("📂 Loading training data...")
    X_train_path = "data/processed/X_train.npy"
    y_train_path = "data/processed/y_train.npy"

    # Check if files exist before loading
    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        raise FileNotFoundError("❌ Processed training data not found. Run `data_load.py` first.")

    # Load the data with allow_pickle=True to handle object arrays
    X_train = np.load(X_train_path, allow_pickle=True)
    y_train = np.load(y_train_path, allow_pickle=True)
    print(f"✅ Training data loaded: {X_train.shape} samples.")

    # Train and save the model
    train_and_save_model(X_train, y_train, args.model_output_path)
