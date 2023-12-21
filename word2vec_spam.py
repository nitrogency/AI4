# Importing necessary libraries for EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords')
 
# Importing libraries necessary for Model Building and Training
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('enron_spam_data.csv') # Replace with location of dataset
data.head()
data.shape

sns.countplot(x='Spam/Ham', data=data) # Replace with header name of Spam/Ham
plt.show()

# Downsampling to balance the dataset

# Replace with header name of Spam/Ham. In this dataset, spam/ham values are saved as 'ham' 'spam' 
# instead of 0 and 1. You might need to change this if your dataset uses those instead.
ham_msg = data[data['Spam/Ham'] == 'ham'] 
spam_msg = data[data['Spam/Ham'] == 'spam']
ham_msg = ham_msg.sample(n=len(spam_msg), replace=True, random_state=42)
 
# Plotting the counts of down sampled dataset
balanced_data = pd.concat([ham_msg, spam_msg], ignore_index=True)
plt.figure(figsize=(8, 6))
sns.countplot(data = balanced_data, x='Spam/Ham') # Replace with header name of Spam/Ham.
plt.title('Distribution of Ham and Spam email messages after downsampling')
plt.xlabel('Message types')
balanced_data['Message'] = balanced_data['Message'].str.replace('Subject', '') # Replace 'Message' with header name where messages are
balanced_data.head()
punctuations_list = string.punctuation
def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

# Apply the remove_punctuations function
balanced_data['Message'].fillna('', inplace=True) # Replace 'Message' with header name where messages are

balanced_data['Message'] = balanced_data['Message'].apply(lambda x: remove_punctuations(x)) # Replace 'Message' with header name where messages are



balanced_data.head()

punctuations_list = string.punctuation
def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

# Apply the remove_punctuations function
balanced_data['Message'].fillna('', inplace=True) # Replace 'Message' with header name where messages are

balanced_data['Message'] = balanced_data['Message'].apply(lambda x: remove_punctuations(x)) # Replace 'Message' with header name where messages are




balanced_data.head()

def remove_stopwords(text):
    stop_words = stopwords.words('english')
 
    imp_words = []
 
    # Storing the important words
    for word in str(text).split():
        word = word.lower()
 
        if word not in stop_words:
            imp_words.append(word)
 
    output = " ".join(imp_words)
 
    return output
 
 
balanced_data['Message'] = balanced_data['Message'].apply(lambda text: remove_stopwords(text)) # Replace 'Message' with header name where messages are
balanced_data.head()
label_mapping = {'spam': 1, 'ham': 0} # This is necessary due to explanation in line 31. Probably 
# not necessary if you're using 0's and 1's.
balanced_data['Spam/Ham'] = balanced_data['Spam/Ham'].map(label_mapping) # Replace with header name of Spam/Ham
#train test split
train_X, test_X, train_Y, test_Y = train_test_split(balanced_data['Message'], # Replace 'Message' with header name where messages are
                                                    balanced_data['Spam/Ham'], # Replace with header name of Spam/Ham
                                                    test_size=0.2,
                                                    random_state=42)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)
 
# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_X)
test_sequences = tokenizer.texts_to_sequences(test_X)
 
# Pad sequences to have the same length
max_len = 100  # maximum sequence length
train_sequences = pad_sequences(train_sequences,
                                maxlen=max_len, 
                                padding='post', 
                                truncating='post')
test_sequences = pad_sequences(test_sequences, 
                               maxlen=max_len, 
                               padding='post', 
                               truncating='post')
# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1,
                                    output_dim=32, 
                                    input_length=max_len))
model.add(tf.keras.layers.LSTM(16))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
 
# Print the model summary
model.summary()
model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
              metrics = ['accuracy'],
              optimizer = 'adam')
es = EarlyStopping(patience=3,
                   monitor = 'val_accuracy',
                   restore_best_weights = True)
 
lr = ReduceLROnPlateau(patience = 2,
                       monitor = 'val_loss',
                       factor = 0.5,
                       verbose = 0)
# Train the model
history = model.fit(train_sequences, train_Y,
                    validation_data=(test_sequences, test_Y),
                    epochs=20, 
                    batch_size=32,
                    callbacks = [lr, es]
                   )

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


