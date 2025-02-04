from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse
from google.cloud import storage
import os

def train_and_save_model(X_train, y_train, model_output_path):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    model_filename = os.path.join(model_output_path, 'model.joblib')
    joblib.dump(model, model_filename)

    # Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket("mlops-bucket")
    blob = bucket.blob("models/model.joblib")
    blob.upload_from_filename(model_filename)
    
    print(f"Model saved to {model_output_path} and uploaded to GCS.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-output-path', dest='model_output_path', default='model', type=str)
    args = parser.parse_args()

    import numpy as np
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")

    train_and_save_model(X_train, y_train, args.model_output_path)
