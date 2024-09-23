# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Step 2: Load the dataset
data = pd.read_csv('Churn_Modelling.csv')

# Step 3: Preprocessing - Encoding categorical variables
label_encoder = LabelEncoder()
data['Geography'] = label_encoder.fit_transform(data['Geography'])
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Step 4: Selecting features and target variable
X = data[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 
          'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
y = data['Exited']

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Model Training - Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_logreg = log_reg.predict(X_test)

# Step 8: Model Training - Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# Step 9: Model Training - Gradient Boosting
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train, y_train)
y_pred_gb = gb_clf.predict(X_test)

# Step 10: Evaluate the models
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))

# Additional Metrics
print("Logistic Regression ROC AUC Score:", roc_auc_score(y_test, log_reg.predict_proba(X_test)[:,1]))
print("Random Forest ROC AUC Score:", roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:,1]))
print("Gradient Boosting ROC AUC Score:", roc_auc_score(y_test, gb_clf.predict_proba(X_test)[:,1]))

# Classification report for detailed evaluation
print("Classification Report - Random Forest:\n", classification_report(y_test, y_pred_rf))
