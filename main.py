# from src.model_training import train_model
# from src.model_evaluation import evaluate_model
# from src.data_preprocessing import load_and_preprocess_data
# from sklearn.model_selection import train_test_split

# if __name__ == "__main__":
#     # Load data and train model
#     X, y = load_and_preprocess_data('data/aid_requests.csv')
#     model = train_model(X, y)
    
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Evaluate the model
#     evaluate_model(model, X_test, y_test)


from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model, save_model_results

def main():
    """Main function to load data, preprocess, train model, and save results."""
    # Load data
    data = load_data('data/new_data.csv')
    
    # Preprocess data
    features, target = preprocess_data(data)
    
    # Train model
    model = train_model(features, target)
    
    # Save results
    save_model_results(model, data)
    
    # Show classified names
    classified_records = data[['Name', 'Predicted Priority Label']]
    print(classified_records)

if __name__ == "__main__":
    main()
