import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

df = pd.read_csv('data/new_data.csv')

print(df.head())