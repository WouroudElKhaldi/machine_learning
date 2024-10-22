import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.pkl')

def preprocess_new_data(data):
    """Preprocess the data to extract features and target."""
    
    # Convert input data to DataFrame
    data_df = pd.DataFrame([data])  # Convert the input dictionary to a DataFrame

    # Parse the Date of Request column
    data_df['Date of Request'] = pd.to_datetime(data_df['Date of Request'], errors='coerce')

    # Extract date features (Day, Month, Year, and Weekday)
    data_df['Day of Request'] = data_df['Date of Request'].dt.day
    data_df['Month of Request'] = data_df['Date of Request'].dt.month
    data_df['Year of Request'] = data_df['Date of Request'].dt.year
    data_df['Is Weekend'] = data_df['Date of Request'].dt.weekday >= 5  # 5 and 6 are Saturday and Sunday

    # Select features and target
    features = data_df.drop(columns=['Request ID', 'Date of Request', 'Contact Info'])  # Removed 'Priority Label'
    
    # One-Hot Encoding for categorical variables, ensuring columns are in the correct order
    features = pd.get_dummies(features, columns=['Name', 'Location', 'Request', 'Category of Aid', 
                                                 'Medical Condition', 'Housing Status', 
                                                 'Urgency Level', 'Current Living Conditions'], drop_first=True)
    
    # Ensure the feature columns are in the same order
    expected_columns = [
        'Name', 'Location', 'Request', 'Category of Aid', 'Medical Condition', 'Number of Dependents', 
        'Income Level', 'Housing Status', 'Urgency Level', 'Day of Request', 'Month of Request', 
        'Year of Request', 'Is Weekend'
    ]
    
    # Add missing columns with 0 if they are not in the dataset
    for col in expected_columns:
        if col not in features.columns:
            features[col] = 0  
    
    features = features[expected_columns]  # Reorder the columns based on expected order
    
    return features

if __name__ == "__main__":
    try:
        # Example new record to predict
        new_record = {
            'Request ID': 4,
            'Name': 'Ali Hassan',
            'Location': 'Tripoli',
            'Request': 'Emergency Medical Assistance',
            'Category of Aid': 'Medical',
            'Medical Condition': 'Chronic Illness',
            'Number of Dependents': 4,
            'Income Level': 100,
            'Housing Status': 'Temporary',
            'Urgency Level': 'High',
            'Date of Request': '10/1/2024',
            'Contact Info': 9617123456,
            'Current Living Conditions': 'Living in a small apartment',
        }
        
        # Preprocess the new record
        new_record_processed = preprocess_new_data(new_record)

        # Make the prediction
        predicted_priority = model.predict(new_record_processed)
        print(f'Predicted Priority Label: {predicted_priority[0]}')

    except FileNotFoundError:
        print("Error: Model file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
