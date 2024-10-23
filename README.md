# Emergency Aid Prioritization using K-Nearest Neighbors (KNN)

## Problem Definition
In Lebanon, the ongoing crisis, including economic hardship and war, has led to an overwhelming number of emergency aid requests from people in dire need. Aid organizations are struggling to process and prioritize these requests efficiently, leading to delays in delivering critical assistance to those most in need. 

This project aims to solve this problem by building a K-Nearest Neighbors (KNN) classifier to prioritize aid requests based on the urgency and severity of the applicants' situations. The goal is to categorize requests into “low priority,” “medium priority,” and “high priority,” ensuring that limited resources are allocated to those who need them most.

## Why Choose This Problem?
The current humanitarian crisis in Lebanon has led to a surge in aid requests, and the efficient allocation of emergency resources is critical for survival. Aid organizations need a fast and reliable way to determine the urgency of each request to provide timely assistance. This project focuses on leveraging machine learning, particularly the KNN algorithm, to provide an accurate, scalable solution to the problem of aid prioritization. 

Using KNN allows us to find similarities between requests based on textual descriptions of the applicants’ needs, ensuring that those who are most similar to previously identified high-priority cases are served first.

## The KNN Solution
We use the KNN algorithm to classify aid requests into priority levels based on various features extracted from the data, including demographic information, descriptions of the applicants' needs, and living conditions. 

The model:
1. **Get the data from the CSV file** simulating real data from Aid Requests.
2. **Converts these textual descriptions** into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency) to represent the importance of words across documents.
3. **Classifies the requests** based on their similarity to previous labeled data (priority levels: low, medium, high).

The KNN classifier compares a new aid request to its nearest neighbors in the dataset to determine its urgency based on similar past entries.

## Data Collection
The dataset includes real-world aid requests collected from various humanitarian organizations. Each request contains:
- **Demographic information** (e.g., name, location, number of dependents, income level).
- **Text descriptions** of the applicant's current needs and conditions (e.g., medical issues, housing status).
- **Urgency labels** that categorize each request into "low priority," "medium priority," and "high priority."

### Example of a Data Record:
```json
{
    "Request ID": 4,
    "Name": "Ali Hassan",
    "Location": "Tripoli",
    "Request": "I need immediate medical care for my chronic asthma that flares up frequently",
    "Category of Aid": "Medical",
    "Medical Condition": "Chronic Illness",
    "Number of Dependents": 4,
    "Income Level": 200,
    "Housing Status": "Temporary",
    "Urgency Level": "High",
    "Date of Request": "2024-10-01",
    "Current Living Conditions": "Living in a small apartment",
}
```

## Steps to Test the Project

1. **Clone the Project**  
   type in the terminal:
   `git clone <repository-url>`

2. **Open the Terminal in VS Code**  
   Ensure the terminal is set to use the Command Prompt.

3. **Set Up a Virtual Environment**  
   `python -m venv venv`  
   `venv\Scripts\activate`

4. **Install the Required Libraries**  
   `pip install -r requirements.txt`
   
6. **Train the Model**  
   Run the following command in the terminal:  
   `python main.py`  
   You should see the terminal return the processed data, which will be saved inside the `classified_data.csv` file.

7. **Test the Model**  
   Open the `test.py` file. You will find a JSON variable called `new_record` that contains the data for prediction. Feel free to adjust the data as needed.  
   To run the test, execute the following command in the terminal:  
   `python test.py`  
   The terminal will display whether the priority is Urgent, Medium, or Low, based on the model's prediction.
