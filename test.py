import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from main import extract_features
from main import preprocess_data

# Load the saved models
rf_model = joblib.load("models/job_recommender.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Load test data
structured_data, unstructured_data = preprocess_data("data/jd_structured_data.csv", "data/jd_unstructured_data.csv")
_, X_structured, _ = extract_features(structured_data, unstructured_data)

y_test = pd.Series([1 if i % 2 == 0 else 0 for i in range(len(X_structured.toarray()))])  # Sample labels

# Predict on test data
y_pred = rf_model.predict(X_structured)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
