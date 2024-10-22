import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data to extract features and target."""
    
    data['Date of Request'] = datee = pd.to_datetime(data['Date of Request'], errors='coerce')
    
    data['Day of Request'] = datee.dt.day
    data['Month of Request'] = datee.dt.month
    data['Year of Request'] = datee.dt.year
    data['Is Weekend'] = datee.dt.weekday >= 5

    features = data.drop(columns=['Priority Label', 'Request ID', 'Date of Request', 'Contact Info'])
    target = data['Priority Label']

    features = pd.get_dummies(features, columns=['Name','Location', 'Request', 'Category of Aid', 'Medical Condition', 
                                                 'Housing Status', 'Urgency Level', 'Current Living Conditions'], drop_first=True)
    
    expected_columns = [
        'Name', 'Location', 'Request', 'Category of Aid', 'Medical Condition', 'Number of Dependents', 
        'Income Level', 'Housing Status', 'Urgency Level', 'Day of Request', 'Month of Request', 'Year of Request', 'Is Weekend'
    ]
    
    for col in expected_columns:
        if col not in features.columns:
            features[col] = 0
    
    features = features[expected_columns]
    
    return features, target

def load_and_preprocess_data(file_path):
    """Load data from a CSV file and preprocess it."""
    data = pd.read_csv(file_path)
    return preprocess_data(data)