import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from tkinter import messagebox
import tkinter as tk
from PIL import Image, ImageTk

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
logistic_model = LogisticRegression(max_iter=500)
logistic_model.fit(X_train, y_train)

# Function to predict spam or ham for user input
def predict_message(model, message):
    # Preprocess the user input
    message = preprocess(message)
    
    # Vectorize the user input
    message_vect = vectorizer.transform([message])
    
    # Predict using the specified model (Naive Bayes or Logistic Regression)
    prediction = model.predict(message_vect)
    
    return prediction[0]

# Tkinter GUI
def check_spam():
    user_message = message_entry.get()
    if not user_message:
        messagebox.showerror("Input Error", "Please enter a message.")
        return

    # Predict using Naive Bayes model
    nb_prediction = predict_message(model, user_message)
    
    # Predict using Logistic Regression model
    logistic_prediction = predict_message(logistic_model, user_message)
    
    # Display predictions
    result_text = f"Naive Bayes Prediction: {nb_prediction}\nLogistic Regression Prediction: {logistic_prediction}"
    result_label.config(text=result_text)

# Function to resize the background image based on window size
def resize_image(event):
    new_width = event.width
    new_height = event.height
    image = original_bg_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    bg_image = ImageTk.PhotoImage(image)
    bg_label.config(image=bg_image)
    bg_label.image = bg_image  # Keep a reference to avoid garbage collection

# Initialize the Tkinter window
window = tk.Tk()
window.title("Spam Detection App")
window.geometry("600x400")  # Set window size

# Load the background image
original_bg_image = Image.open("spam msg bg.jpg")

# Convert to PhotoImage
bg_image = ImageTk.PhotoImage(original_bg_image)

# Create a label for the background image
bg_label = tk.Label(window, image=bg_image)
bg_label.place(relwidth=1, relheight=1)

# Bind the window resizing event to resize the background image
window.bind("<Configure>", resize_image)

# Add a heading label
heading_label = tk.Label(window, text="Spam Detection System", font=("Helvetica", 18, "bold"), bg="#ffcccb", fg="black")
heading_label.place(relx=0.5, rely=0.1, anchor="center")

# Create and place widgets with proper spacing and alignment
# Create and place widgets with proper spacing and alignment
message_label = tk.Label(window, text="Enter a message:", font=("Helvetica", 12), bg="#ffcccb", fg="black")
message_label.place(relx=0.5, rely=0.3, anchor="center")  # Centered horizontally

message_entry = tk.Entry(window, width=30, font=("Helvetica", 12))
message_entry.place(relx=0.5, rely=0.5, anchor="center")  # Centered horizontally


check_button = tk.Button(window, text="Check", command=check_spam, font=("Helvetica", 12), bg="#ff9999", fg="black")
check_button.place(relx=0.5, rely=0.7, anchor="center")

result_label = tk.Label(window, text="", font=("Helvetica", 12), bg="#ffcccb", fg="black")
result_label.place(relx=0.5, rely=0.6, anchor="center")

# Start the Tkinter event loop
window.mainloop()
