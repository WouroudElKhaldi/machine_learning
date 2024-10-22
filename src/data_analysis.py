import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_results(file_path):
    """Analyze the results from the classified data."""
    data = pd.read_csv(file_path)
    
    priority_order = ['Urgent', 'Medium', 'Low']
    
    data['Predicted Priority Label'] = pd.Categorical(data['Predicted Priority Label'], categories=priority_order, ordered=True)
    
    sorted_data = data.sort_values(by='Predicted Priority Label', ascending=True, key=lambda x: x.cat.codes)

    # Create the count plot
    sns.countplot(x='Predicted Priority Label', data=sorted_data, order=priority_order)
    plt.title('Count of Requests by Predicted Priority')
    plt.show()

    return sorted_data

if __name__ == "__main__":
    sorted_results = analyze_results('classified_data.csv')
    print(sorted_results)
