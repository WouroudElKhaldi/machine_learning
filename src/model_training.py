import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from src.data_preprocessing import preprocess_data
import matplotlib.pyplot as plt
import numpy as np
import joblib

def train_model(features, target):
    """Train a Random Forest model on the provided features and target."""

    data = pd.read_csv('data/new_data.csv')

    data['Date of Request'] = datee = pd.to_datetime(data['Date of Request'], errors='coerce')

    data['Year of Request'] = datee.dt.year
    data['Month of Request'] = datee.dt.month
    data['Day of Request'] = datee.dt.day
    
    features = data.drop(columns=['Date of Request'])
    
    data = data.dropna(subset=['Priority Label'])

    features_encoded, _ = preprocess_data(data)
    
    X = features_encoded
    y = data['Priority Label']

    print(f"Shape of features (X): {X.shape}")
    print(f"Shape of target (y): {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, 'model.pkl')
    
    predictions = model.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    
    return model


def save_model_results(model, data):
    """Save the model's predictions alongside the original data."""
    
    features_encoded, _ = preprocess_data(data)
    
    data['Predicted Priority Label'] = model.predict(features_encoded)
    
    data.to_csv('classified_data.csv', index=False)
    print("Classified data saved to 'classified_data.csv'.")
    
    joblib.dump(model, 'model.pkl')
    print("Model saved to 'model.pkl'.")

def plot_feature_importance(model, features):
    """Plot the feature importances from the trained model."""
    importances = model.feature_importances_
    feature_names = features.columns
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(features.shape[1]), importances[indices], align="center")
    plt.xticks(range(features.shape[1]), feature_names[indices], rotation=90)
    plt.xlim([-1, features.shape[1]])
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv('data/new_data.csv')
    features = data.drop(columns=['Priority Label']).columns
    target = data['Priority Label']
    model = train_model(features, target)
    save_model_results(model, data)
    plot_feature_importance(model, data[features])
