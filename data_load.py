import pandas as pd
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Get the current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df_path = os.path.join(BASE_DIR, "sales_data.csv")  # Relative path

# Load Data
if not os.path.exists(df_path):
    raise FileNotFoundError(f"Dataset not found at {df_path}. Ensure the file is in the repository.")

df = pd.read_csv(df_path)

# Define Features (X) and Target (y)
X = df.drop('Units_Sold', axis=1)
y = df['Units_Sold']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Column Transformer for Preprocessing
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numeric_features)]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save preprocessor
preprocessor_path = os.path.join(BASE_DIR, 'preprocessor.joblib')
joblib.dump(preprocessor, preprocessor_path)

# Save Processed Data
np.save(os.path.join(BASE_DIR, 'X_train.npy'), X_train_processed)
np.save(os.path.join(BASE_DIR, 'y_train.npy'), y_train.values)
np.save(os.path.join(BASE_DIR, 'X_test.npy'), X_test_processed)
np.save(os.path.join(BASE_DIR, 'y_test.npy'), y_test.values)

print("Data preprocessing complete.")