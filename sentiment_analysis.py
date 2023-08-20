import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import movie_reviews

# Download the movie reviews dataset from NLTK
nltk.download('movie_reviews')
nltk.download('punkt')

# Load positive and negative reviews
positive_reviews = [(review, 'positive') for review in movie_reviews.sents(categories='pos')]
negative_reviews = [(review, 'negative') for review in movie_reviews.sents(categories='neg')]
all_reviews = positive_reviews + negative_reviews

# Shuffle the reviews
np.random.shuffle(all_reviews)

# Split the dataset into features and labels
X = [" ".join(review) for (review, sentiment) in all_reviews]
y = [sentiment for (review, sentiment) in all_reviews]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CountVectorizer to convert text data into numerical features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Create a Naive Bayes classifier
clf = MultinomialNB()

# Train the classifier on the training data
clf.fit(X_train_vec, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
