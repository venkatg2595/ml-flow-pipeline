import os
import json
import pandas as pd
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

# Replace with your GCS bucket and file name
BUCKET_NAME = "mlops-bucket12"
FILE_NAME = "titanic_data.csv"
GCS_PATH = f"gs://{BUCKET_NAME}/{FILE_NAME}"

print(f"📂 Loading data from: {GCS_PATH}")

# Read CSV file from GCS
df = pd.read_csv(GCS_PATH)

# ----------------------------
# ✅ Preprocessing
# ----------------------------

df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill missing values
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df.drop(['embarked'], axis=1, inplace=True)

X = df.drop('survived', axis=1)
y = df['survived']

# Feature engineering (example)
X['FamilySize'] = X['sibsp'] + X['parch'] + 1

# ----------------------------
# ✅ Save Processed Data to BigQuery
# ----------------------------

DATASET_ID = "mlops_dataset"  # Replace with your dataset
TABLE_NAME = "titanic_data"
TABLE_ID = f"{client.project}.{DATASET_ID}.{TABLE_NAME}"

print(f"📡 Uploading to BigQuery table: {TABLE_ID}")

# Upload DataFrame to BigQuery
job = client.load_table_from_dataframe(df, TABLE_ID)
job.result()  # Wait for job to complete

print("✅ Data successfully uploaded to BigQuery!")

