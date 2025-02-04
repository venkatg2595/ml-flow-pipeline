from google.cloud import bigquery
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Initialize BigQuery client
client = bigquery.Client()

# Query data from BigQuery
query = "SELECT * FROM `your-project-id.mlops_dataset.titanic_data`"
df = client.query(query).to_dataframe()

# Preprocessing
df.fillna(df.mean(numeric_only=True), inplace=True)  
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

X = df.drop(columns=['survived'])
y = df['survived']

# Feature engineering
X['FamilySize'] = X['sibsp'] + X['parch'] + 1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessor
joblib.dump(scaler, "model/preprocessor.joblib")

print("Data preprocessing complete.")
