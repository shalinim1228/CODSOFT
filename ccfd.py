# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the datasets
fraud_train = pd.read_csv('credit_card_fraud_detection/fraudTrain.csv')
fraud_test = pd.read_csv('credit_card_fraud_detection/fraudTest.csv')

# Step 2: Data Preprocessing
# Convert 'trans_date_trans_time' and 'dob' to datetime
fraud_train['trans_date_trans_time'] = pd.to_datetime(fraud_train['trans_date_trans_time'])
fraud_train['dob'] = pd.to_datetime(fraud_train['dob'])

fraud_test['trans_date_trans_time'] = pd.to_datetime(fraud_test['trans_date_trans_time'])
fraud_test['dob'] = pd.to_datetime(fraud_test['dob'])

# Create new features: 'transaction_hour' and 'age'
fraud_train['transaction_hour'] = fraud_train['trans_date_trans_time'].dt.hour
fraud_train['age'] = (fraud_train['trans_date_trans_time'] - fraud_train['dob']).dt.days // 365

fraud_test['transaction_hour'] = fraud_test['trans_date_trans_time'].dt.hour
fraud_test['age'] = (fraud_test['trans_date_trans_time'] - fraud_test['dob']).dt.days // 365

# Drop unnecessary columns
cols_to_drop = ['trans_date_trans_time', 'dob', 'first', 'last', 'street', 'city', 'state', 'zip', 'job', 'trans_num']
fraud_train = fraud_train.drop(columns=cols_to_drop)
fraud_test = fraud_test.drop(columns=cols_to_drop)

# Encode categorical variables (merchant, category, gender)
label_enc = LabelEncoder()
fraud_train['merchant'] = label_enc.fit_transform(fraud_train['merchant'])
fraud_train['category'] = label_enc.fit_transform(fraud_train['category'])
fraud_train['gender'] = label_enc.fit_transform(fraud_train['gender'])

# Transform test data using the already fitted label encoder
try:
    fraud_test['merchant'] = label_enc.transform(fraud_test['merchant'])
except KeyError as e:
    print(f"KeyError: {e}. Handling unseen labels.")
    fraud_test['merchant'] = fraud_test['merchant'].apply(
        lambda x: label_enc.transform([x])[0] if x in label_enc.classes_ else -1
    )

fraud_test['category'] = label_enc.transform(fraud_test['category'])
fraud_test['gender'] = label_enc.transform(fraud_test['gender'])

# Step 3: Define features (X) and target (y)
X_train = fraud_train.drop(columns=['is_fraud'])
y_train = fraud_train['is_fraud']

X_test = fraud_test.drop(columns=['is_fraud'])
y_test = fraud_test['is_fraud']

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Modeling

# Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)

# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Step 5: Evaluation

# Logistic Regression Results
print("Logistic Regression Performance:")
print(f'Accuracy: {accuracy_score(y_test, y_pred_lr):.2f}')
print(classification_report(y_test, y_pred_lr))

# Decision Tree Results
print("\nDecision Tree Performance:")
print(f'Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}')
print(classification_report(y_test, y_pred_dt))

# Random Forest Results
print("\nRandom Forest Performance:")
print(f'Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}')
print(classification_report(y_test, y_pred_rf))

# Step 6: Plotting Confusion Matrix for Random Forest Model
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Random Forest')
plt.show()
