#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


# Read CSV file

# In[2]:


df = pd.read_csv (r'IMDB.csv')
print (df)


# Preprocessing

# In[3]:


# Function to remove punctuation from a single text string
def remove_punctuation(review):
    # Use regex to replace punctuation with an empty string
    return ''.join([char for char in review if char not in string.punctuation])

# Apply the function to the 'Text' column in the dataframe
df['review'] = df['review'].apply(remove_punctuation)

# Display the cleaned dataframe
print(df)


# In[4]:


# Convert the 'Review' column to lowercase
df['review'] = df['review'].str.lower()

# Display the dataframe with lowercase text
print(df)


# In[5]:


# Function to remove special characters and numbers using regex
def remove_special_characters_and_numbers(review):
    # Use regex to remove any characters that are not letters or whitespace
    return re.sub(r'[^a-zA-Z\s]', '', review)

# Apply the function to the 'Text' column in the dataframe
df['review'] = df['review'].apply(remove_special_characters_and_numbers)

# Display the cleaned dataframe
print(df)


# In[6]:


nltk.download('stopwords')
nltk.download('punkt')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(review):
    tokens = nltk.word_tokenize(review)
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [stemmer.stem(word) for word in tokens]  # Perform stemming
    return ' '.join(tokens)

df['review'] = df['review'].apply(preprocess_text)
print(df)



# Feature Extraction & Model Selection

# In[7]:


X = df['review']  # Features (text)
y = df['sentiment']   # sentiment

# Create a CountVectorizer for Bag-of-Words
count_vectorizer = CountVectorizer(max_features=5000)  # Set the desired max_features
X_bow = count_vectorizer.fit_transform(y)

# Print the Bag-of-Words representation
print("Bag-of-Words (BoW) representation:")
print(X_bow.toarray())


# In[8]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Set the desired max_features

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

logistic_regression = LogisticRegression(C=1.0)  # Adjust C as needed
logistic_regression.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = logistic_regression.predict(X_test_tfidf)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display a classification report with precision, recall, and F1-score
classification_rep = classification_report(y_test, y_pred)
print("\nClassification Report:\n", classification_rep)


# Training

# In[9]:


vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

y_pred = nb_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

print(classification_report(y_test, y_pred))


# Testing

# In[10]:


new_reviews = [
    "Good movie i loved it",
    "Very bad and boring",
    # Add more new reviews here
]

new_reviews_tfidf = vectorizer.transform(new_reviews)
new_reviews_predictions = nb_classifier.predict(new_reviews_tfidf)
for i, review in enumerate(new_reviews):
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {new_reviews_predictions[i]}")
    # Compare with the true sentiment if available


# Evaluation

# In[11]:


# Assuming 'y_test' contains your testing labels and 'y_pred' contains your model's predictions

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1-score for both classes
precision_negative = precision_score(y_test, y_pred, average=None)[0]
recall_negative = recall_score(y_test, y_pred, average=None)[0]
f1_negative = f1_score(y_test, y_pred, average=None)[0]

precision_positive = precision_score(y_test, y_pred, average=None)[1]
recall_positive = recall_score(y_test, y_pred, average=None)[1]
f1_positive = f1_score(y_test, y_pred, average=None)[1]

# Display the results
print("Accuracy:", accuracy)
print("Precision (Negative class):", precision_negative)
print("Recall (Negative class):", recall_negative)
print("F1-score (Negative class):", f1_negative)
print("Precision (Positive class):", precision_positive)
print("Recall (Positive class):", recall_positive)
print("F1-score (Positive class):", f1_positive)

# Display a full classification report
classification_rep = classification_report(y_test, y_pred)
print("\nClassification Report:\n", classification_rep)


#  simple text-based interface

# In[12]:


# Function to predict sentiment
def predict_sentiment(review_text):
    review_text = review_text.lower()
    review_text = re.sub('[^a-zA-Z\s]', '', review_text)
    review_tfidf = vectorizer.transform([review_text])
    
    # Predict the probabilities for both classes (0 for negative and 1 for positive)
    probabilities = nb_classifier.predict_proba(review_tfidf)
    
    # Get the index of the class with the higher probability
    predicted_class = int(probabilities.argmax())
    
    return "Positive" if predicted_class == 1 else "Negative"


# In[ ]:


# Main interface loop
while True:
    user_input = input("Enter a movie review (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    sentiment = predict_sentiment(user_input)
    print("Predicted Sentiment:", sentiment)


# In[ ]:




