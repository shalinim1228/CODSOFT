1. Loading the Dataset:
The dataset is loaded from a CSV file containing labeled messages. The two relevant columns are label (which indicates whether the message is spam or ham) and message (which contains the text of the message). The dataset is encoded using ISO-8859-1 to handle special characters.

2. Preprocessing the Text:
To prepare the text data for model training, several preprocessing steps are applied:

Lowercasing: All text is converted to lowercase to ensure consistency.
Removing Punctuation: Punctuation marks are removed to focus solely on the words.
Stopword Removal: Common stopwords (e.g., "the", "is") are removed using the NLTK library to reduce noise and improve model focus on meaningful words.
3. TF-IDF Vectorization:
A TfidfVectorizer is used to convert the text messages into numerical features. This technique calculates Term Frequency-Inverse Document Frequency (TF-IDF) scores for each word, emphasizing important words in the message and down-weighting frequently occurring words across many messages.

4. Train-Test Split:
The data is split into training and testing sets, with 80% of the data used for training the models and 20% reserved for testing. This ensures that the models can be evaluated on unseen data to assess their performance.

5. Training Models:
Two models are trained:

Naive Bayes (MultinomialNB): This model is well-suited for text classification tasks, especially when dealing with word frequencies. It assumes independence between features (words in this case).
Logistic Regression: A more general model for binary classification, Logistic Regression finds a decision boundary between spam and ham messages.
6. Model Evaluation:
Both models are evaluated on the test data:

Accuracy Score: This metric shows the percentage of correct predictions.
Classification Report: This includes precision, recall, and F1-score, providing deeper insights into how well the models perform in predicting both spam and ham.
7. Prediction Function for User Input:
The script includes a function that allows users to input a message and predict whether it is spam or ham. The input message is preprocessed and vectorized in the same way as the training data, and the prediction is made using either the Naive Bayes or Logistic Regression model.

8. User Interaction:
The script prompts the user to input a message, and the message is classified as spam or ham using both the Naive Bayes and Logistic Regression models. The predictions from both models are printed for comparison.

This script provides a robust pipeline for spam detection, from data preprocessing to model evaluation and real-time user input prediction.
