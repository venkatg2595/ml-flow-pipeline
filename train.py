from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import argparse
import numpy as np

def train_and_save_model(X_train, y_train, model_output_path):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Ensure output directory exists
    os.makedirs(model_output_path, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_output_path, 'regression_model.joblib')
    joblib.dump(model, model_path)
    
    print(f"Regression model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-output-path', dest='model_output_path', 
                        default=os.path.join(os.path.dirname(__file__), 'model'), type=str, 
                        help='Path to save trained model')
    args = parser.parse_args()

    X_train = np.load(os.path.join(os.path.dirname(__file__), 'X_train.npy'))
    y_train = np.load(os.path.join(os.path.dirname(__file__), 'y_train.npy'))

    train_and_save_model(X_train, y_train, args.model_output_path)