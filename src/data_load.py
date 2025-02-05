import os
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account

# ----------------------------
# ✅ Google Cloud Authentication
# ----------------------------

# Load credentials from environment variable
gcp_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not gcp_credentials:
    raise ValueError("❌ Google Cloud credentials not found. Make sure GOOGLE_APPLICATION_CREDENTIALS is set.")

credentials = service_account.Credentials.from_service_account_file(gcp_credentials)
client = bigquery.Client(credentials=credentials)

# ----------------------------
# ✅ Load Data from GCS
# ----------------------------

BUCKET_NAME = "mlops-bucket12"
FILE_NAME = "titanic.csv"
GCS_PATH = f"gs://{BUCKET_NAME}/{FILE_NAME}"

print(f"📂 Loading data from: {GCS_PATH}")

try:
    df = pd.read_csv(GCS_PATH)
    print("✅ File loaded successfully.")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit(1)

print("🔍 Available Columns:", df.columns.tolist())

# ----------------------------
# ✅ Preprocessing
# ----------------------------

df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill missing values

if 'sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
else:
    print("⚠️ Column 'Sex' not found. Skipping mapping.")

if 'embarked' in df.columns:
    df.drop(['embarked'], axis=1, inplace=True)
    print("✅ Dropped column: 'embarked'")
else:
    print("⚠️ Column 'embarked' not found. Skipping drop operation.")

if 'Survived' in df.columns:
    X = df.drop('Survived', axis=1)
    y = df['Survived']
else:
    print("❌ Column 'Survived' not found. Check dataset!")
    exit(1)

# Feature engineering
if 'sibsp' in X.columns and 'parch' in X.columns:
    X['FamilySize'] = X['sibsp'] + X['parch'] + 1
    print("✅ Created 'FamilySize' feature.")

# ----------------------------
# ✅ Save Processed Data
# ----------------------------

processed_dir = "data/processed"
os.makedirs(processed_dir, exist_ok=True)

# Save processed features and labels
np.save(os.path.join(processed_dir, "X_train.npy"), X)
np.save(os.path.join(processed_dir, "y_train.npy"), y)

print("✅ Processed data saved successfully!")

# ----------------------------
# ✅ Save Data to BigQuery
# ----------------------------

DATASET_ID = "mlops_dataset"
TABLE_NAME = "titanic_data"
TABLE_ID = f"{client.project}.{DATASET_ID}.{TABLE_NAME}"

print(f"📡 Uploading to BigQuery table: {TABLE_ID}")

try:
    job = client.load_table_from_dataframe(df, TABLE_ID)
    job.result()  # Wait for job to complete
    print("✅ Data successfully uploaded to BigQuery!")
except Exception as e:
    print(f"❌ Error uploading to BigQuery: {e}")
