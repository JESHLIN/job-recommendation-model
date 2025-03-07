import spacy
import pdfplumber
import pandas as pd
import re

csv_path = "data/skills.csv"  # Ensure the correct path

try:
    skills_df = pd.read_csv(csv_path)
    print("CSV Loaded Successfully")
    print("Column Names:", skills_df.columns)
    print("First 5 Rows:\n", skills_df.head())
except Exception as e:
    print("Error loading CSV:", str(e))

def load_skills(csv_path="data/skills.csv"):
    """Loads predefined skills from a CSV file and ensures correct column selection."""
    skills_df = pd.read_csv(csv_path)

    # Debug: Print available columns
    print("CSV Columns:", skills_df.columns)

    # Auto-detect the column containing skills
    possible_columns = ["technical skills", "skills", "Skill", "skill", "expertise"]
    skill_column = next((col for col in possible_columns if col in skills_df.columns), None)

    if skill_column is None:
        raise ValueError("No valid skill column found in skills.csv. Check the column names.")

    skills_list = skills_df[skill_column].dropna().str.lower().str.strip().tolist()
    print(f"Loaded {len(skills_list)} skills from CSV.")

    return set(skills_list)


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    extracted_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)

    full_text = "\n".join(extracted_text)
    
    if not full_text.strip():
        raise ValueError("No text extracted from PDF. Check if it's a scanned document.")
    
    return full_text

def clean_text(text):
    """Preprocesses text by removing special characters and extra spaces."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()

def extract_skills_from_text(text, predefined_skills):
    """Extracts skills from text based on a predefined skill list."""
    extracted_skills = {skill for skill in predefined_skills if skill in text}
    
    if not extracted_skills:
        print("No matching skills found. Check the skill list and text processing.")
    
    return list(extracted_skills)

def parse_resume(pdf_path, skills_csv="data/skills.csv"):
    """Parses a resume PDF and extracts relevant skills."""
    predefined_skills = load_skills(skills_csv)
    text = extract_text_from_pdf(pdf_path)
    clean_resume_text = clean_text(text)
    extracted_skills = extract_skills_from_text(clean_resume_text, predefined_skills)
    
    return {"skills": extracted_skills, "text": clean_resume_text}

if __name__ == "__main__":
    pdf_resume = "utilities/resumes/CV.pdf"  # Replace with actual path
    parsed_data = parse_resume(pdf_resume)
    
    print("\n Extracted Skills:", parsed_data["skills"])
    print("\n Extracted Resume Text (Preview):\n", parsed_data["text"][:500])  # Show first 500 chars