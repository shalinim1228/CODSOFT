import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

# Load your CSV file
data = pd.read_csv('spam.csv', usecols=[0, 1], names=['label', 'message'], skiprows=1, encoding='ISO-8859-1')

# Preprocessing
nltk.download('stopwords')

def preprocess(text):
    text = text.lower()  # Lowercase the text
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

# Apply preprocessing to the messages
data['message'] = data['message'].apply(preprocess)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Model evaluation (optional)
y_pred_nb = model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

y_pred_logistic = logistic_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic))

# Function to predict spam or ham for user input
def predict_message(model, message):
    # Preprocess the user input
    message = preprocess(message)
    
    # Vectorize the user input
    message_vect = vectorizer.transform([message])
    
    # Predict using the specified model (Naive Bayes or Logistic Regression)
    prediction = model.predict(message_vect)
    
    return prediction[0]

# Get user input
user_message = input("Enter a message to check if it's spam or ham: ")

# Predict using Naive Bayes model
nb_prediction = predict_message(model, user_message)
print(f"Naive Bayes Prediction: {nb_prediction}")

# Predict using Logistic Regression model
logistic_prediction = predict_message(logistic_model, user_message)
print(f"Logistic Regression Prediction: {logistic_prediction}")
