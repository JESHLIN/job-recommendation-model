import os
import json
import pandas as pd
import spacy
import numpy as np
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

# Ensure the spaCy model is installed
def load_spacy_model():
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Model '{model_name}' not found. Downloading now...")
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        return spacy.load(model_name)

nlp = load_spacy_model()

# Weightages
talent_weights = {
    "Skills": 0.24,
    "Experience": 0.20,
    "Industry Fit": 0.16,
    "Portfolio": 0.08,
    "Willingness": 0.08,
    "Feedback": 0.04,
    "Hourly Rate": 0.20,
}

business_weights = {
    "Skills": 0.20,
    "Experience": 0.16,
    "Industry Fit": 0.24,
    "Feedback": 0.12,
    "Business Growth": 0.08,
    "Hourly Rate": 0.20,
}

# Preprocess text
def preprocess_text(text):
    """Cleans and preprocesses text data."""
    if pd.isna(text):
        return ""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# Load and preprocess data
def preprocess_data(structured_path, unstructured_path):
    structured_data = pd.read_csv(structured_path)
    unstructured_data = pd.read_csv(unstructured_path)

    structured_data["Processed_JD"] = structured_data["Processed_JD"].apply(preprocess_text)
    unstructured_data["Job Description"] = unstructured_data["Job Description"].apply(preprocess_text)

    return structured_data, unstructured_data

# Feature Engineering
def extract_features(structured_data, unstructured_data):
    vectorizer = TfidfVectorizer()
    X_structured = vectorizer.fit_transform(structured_data["Processed_JD"])
    X_unstructured = vectorizer.transform(unstructured_data["Job Description"])
    return vectorizer, X_structured, X_unstructured

# Train Machine Learning Model
def train_models(X_structured):
    """Trains the Random Forest model with corrected label handling."""
    num_samples = X_structured.shape[0]  # Correct way to get number of rows
    y_labels = np.random.randint(0, 2, size=num_samples)  # Generate labels

    X_train, X_test, y_train, y_test = train_test_split(X_structured, y_labels, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)

    print("Random Forest Model - Precision:", precision_score(y_test, y_pred))
    print("Random Forest Model - Recall:", recall_score(y_test, y_pred))
    print("Random Forest Model - F1 Score:", f1_score(y_test, y_pred))

    return rf_model

# Compute Job Match Score
def compute_match_score(candidate, job):
    """Computes a weighted match score for a candidate and job."""
    score = sum(talent_weights.get(k, 0) * candidate.get(k, 0) for k in talent_weights)
    score += sum(business_weights.get(k, 0) * job.get(k, 0) for k in business_weights)
    return score

# Resume Parsing
def parse_resume(resume_text):
    doc = nlp(resume_text)
    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    return {"skills": skills, "text": resume_text}

# Save Trained Models
def save_models(rf_model, vectorizer):
    joblib.dump(rf_model, "models/job_recommender.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("Models saved successfully!")

# Main Script
def main():
    structured_data, unstructured_data = preprocess_data("data/jd_structured_data.csv", "data/jd_unstructured_data.csv")
    vectorizer, X_structured, X_unstructured = extract_features(structured_data, unstructured_data)
    rf_model = train_models(X_structured)
    save_models(rf_model, vectorizer)

if __name__ == "__main__":
    main()
