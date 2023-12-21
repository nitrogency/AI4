import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

spam = pd.read_csv('enron_spam_data.csv') # Replace with location of dataset
spam['Message'].fillna('', inplace=True) # Replace 'Message' with header name where messages are
z = spam['Message'] # Replace 'Message' with header name where messages are
y = spam["Spam/Ham"] # Replace with header name of Spam/Ham
z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2)

cv = CountVectorizer()
features = cv.fit_transform(z_train)

model = svm.SVC()
model.fit(features,y_train)

features_test = cv.transform(z_test)
print(model.score(features_test,y_test))
