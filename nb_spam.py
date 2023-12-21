
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
import string
from pickle import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
nltk.download('stopwords')
# Load the dataset
dataset = pd.read_csv('enron_spam_data.csv') # Replace with location of dataset
dataset.shape
# Show dataset head (first 5 records)
dataset.head() 
# Show dataset info
dataset.info()
# Show dataset statistics
dataset.describe()
# Visualize spam  frequenices
plt.figure(dpi=100)
sns.countplot(dataset['Spam/Ham']) # Replace with header name of Spam/Ham
plt.title("Spam Freqencies")
plt.show()
# Check for missing data for each column 
dataset.isnull().sum()
# Check for duplicates and remove them 
dataset.drop_duplicates(inplace=True)
# Cleaning data from punctuation and stopwords and then tokenizing it into words (tokens)
def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean
# Fit the CountVectorizer to data
dataset['Message'].fillna('', inplace=True) # Replace 'Message' with header name where messages are
message = CountVectorizer(analyzer=process).fit_transform(dataset['Message']) # Replace 'Message' with header name where messages are
# Save the vectorizer
dump(message, open("vectorizer.pkl", "wb"))
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(message, dataset['Spam/Ham'], test_size=0.20, random_state=0) # Replace with header name of Spam/Ham
# Model creation
model = MultinomialNB()
# Model training
model.fit(X_train, y_train)
# Model saving
dump(model, open("model.pkl", 'wb'))
# Model predictions on test set
y_pred = model.predict(X_test)
# Model Evaluation | Accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy * 100
# Model Evaluation | Classification report
classification_report(y_test, y_pred)
# Model Evaluation | Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(dpi=100)
sns.heatmap(cm, annot=True)
plt.title("Confusion matrix")
plt.show()
