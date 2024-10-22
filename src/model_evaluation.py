from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from model_training import train_model
from data_preprocessing import load_and_preprocess_data

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using confusion matrix and classification report."""
    predictions = model.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    X, y = load_and_preprocess_data('data/new_data.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    
    evaluate_model(model, X_test, y_test)

