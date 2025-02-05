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
    
    # Initialize the RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    
    # Train the model with the processed data
    model.fit(X_train, y_train)

    # Ensure output directory exists
    os.makedirs(model_output_path, exist_ok=True)

    # Save the trained model locally as a .joblib file
    model_filename = os.path.join(model_output_path, 'model.joblib')
    joblib.dump(model, model_filename)
    print(f"✅ Model saved locally: {model_filename}")

    # Upload the trained model to Google Cloud Storage (GCS)
    upload_to_gcs(model_filename, "mlops-bucket", "models/model.joblib")

# ----------------------------
# ✅ Upload Model to GCS
# ----------------------------
def upload_to_gcs(local_file, bucket_name, gcs_path):
    print("📡 Uploading model to Google Cloud Storage...")
    
    # Initialize the GCS client
    storage_client = storage.Client()
    
    # Access the bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Create a blob for the model path
    blob = bucket.blob(gcs_path)
    
    # Upload the model file
    blob.upload_from_filename(local_file)
    print(f"✅ Model uploaded to GCS: gs://{bucket_name}/{gcs_path}")

# ----------------------------
# ✅ Main Execution
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-output-path', dest='model_output_path', default='model', type=str)
    args = parser.parse_args()

    # Load preprocessed training data (X_train and y_train)
    print("📂 Loading training data...")

    X_train_path = "data/processed/X_train.npy"
    y_train_path = "data/processed/y_train.npy"

    # Check if processed files exist before loading
    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        raise FileNotFoundError("❌ Processed training data not found. Run `data_load.py` first.")

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    print(f"✅ Training data loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features.")

    # Train the model and save it
    train_and_save_model(X_train, y_train, args.model_output_path)
