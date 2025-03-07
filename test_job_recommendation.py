import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from resume_parser import parse_resume, clean_text

# Load trained models
vectorizer = joblib.load("models/vectorizer.pkl")
rf_model = joblib.load("models/job_recommender.pkl")

# Load job data
job_data = pd.read_csv("data/jd_structured_data.csv")

# Ensure required columns exist
required_columns = ["Job Title", "Job Description"]
for col in required_columns:
    if col not in job_data.columns:
        print(f"Warning: Missing column '{col}' in job data.")
        job_data[col] = "Unknown"

# Drop empty job descriptions
job_data = job_data.dropna(subset=["Job Description"])

# Preprocess job descriptions
job_data["Job Description"] = job_data["Job Description"].astype(str).apply(clean_text)

# Parse test resume
resume_path = "utilities/resumes/CV.pdf"  # Update path if necessary
parsed_resume = parse_resume(resume_path)
resume_text = clean_text(parsed_resume["text"])

# Print extracted resume text to debug
print("Extracted Resume Text (First 500 characters):\n", resume_text[:500])

# Vectorize resume
resume_vector = vectorizer.transform([resume_text])

# Vectorize job descriptions
job_vectors = vectorizer.transform(job_data["Job Description"])

# Compute similarity scores
similarity_scores = cosine_similarity(resume_vector, job_vectors).flatten()

def get_job_recommendations(similarity_scores, job_data, top_n=5):
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]  # Get top N indices
    recommendations = job_data.iloc[top_indices][["Job Title", "Job Description"]].copy()
    #recommendations["Similarity Score"] = similarity_scores[top_indices]
    return recommendations

# Get recommendations
recommendations = get_job_recommendations(similarity_scores, job_data)

print("Top Job Recommendations:")
print(recommendations)
