import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
from pathlib import Path

# Paths
MODEL_DIR = Path(r"C:\A EVELYN YEOH\Sunway Uni\Y3S3\Machine Learning\Assignment\model")
TFIDF     = joblib.load(MODEL_DIR / "tfidf.pkl")
TEXT_MOD  = joblib.load(MODEL_DIR / "text_model.pkl")
META_MOD  = joblib.load(MODEL_DIR / "meta_model.pkl")

# Constants
STOPWORDS = {'a','an','the','and','or','but','if','while','is','are','was','were',
             'be','been','being','have','has','had','do','does','did','for','of',
             'in','on','at','to','from','by','with','about','as','into','like',
             'through','after','over','between','out','against','during','without',
             'before','under','around','among'}

RED_KEYWORDS = ["quick money","fast hiring","unlimited income","work from anywhere",
                "no experience","immediate start","commission only","urgent requirement",
                "quick apply","get rich"]

# Helper functions
def clean_text(txt):
    txt = re.sub(r"\d+", " ", txt.lower())
    txt = re.sub(r"[^\w\s]", " ", txt)
    return " ".join(w for w in txt.split() if w not in STOPWORDS)

def engineer_meta(df):
    df = df.copy()
    df["salary_log"]      = np.log1p(df["monthly_salary"].clip(lower=0))
    df["low_salary_flag"] = (df["monthly_salary"] < 1000).astype(int)
    df["is_remote_loc"]   = df["location"].str.contains(r"remote|anywhere", case=False, na=False).astype(int)
    return df

def predict_fake_job(job_post):
    # Text preprocessing
    raw = " ".join(job_post.get(k, "") for k in ["title", "company_profile", "description", "requirements"])
    clean = clean_text(raw)
    v_tfidf = TFIDF.transform([clean])
    v_num   = csr_matrix([[len(clean), clean.count("!"),
                           sum(1 for w in clean.split() if w.isupper() and len(w) > 1),
                           sum(1 for k in RED_KEYWORDS if k in clean)]])
    X_text_new = hstack([v_tfidf, v_num])
    p_text = TEXT_MOD.predict_proba(X_text_new)[0][1]

    # Metadata
    meta_df = engineer_meta(pd.DataFrame([job_post]))
    p_meta  = META_MOD.predict_proba(meta_df)[0][1]

    # Final blend
    p_final = 0.6 * p_text + 0.4 * p_meta
    if any(k in clean for k in RED_KEYWORDS):
        p_final = max(p_final, 0.9)
    
    return p_final

# Streamlit UI
st.title("🕵️‍♀️ Fake Job Detector AI")

with st.form("job_form"):
    title = st.text_input("Job Title")
    company_profile = st.text_area("Company Profile")
    description = st.text_area("Job Description")
    requirements = st.text_area("Job Requirements")
    location = st.text_input("Job Location")
    department = st.text_input("Department")
    salary = st.number_input("Monthly Salary", min_value=0.0, value=3000.0, step=100.0)

    submitted = st.form_submit_button("Predict")

    if submitted:
        job_post = {
            "title": title,
            "company_profile": company_profile,
            "description": description,
            "requirements": requirements,
            "location": location,
            "department": department,
            "monthly_salary": salary
        }

        prob = predict_fake_job(job_post)
        st.markdown("---")
        st.subheader("🔍 Prediction Result")
        st.write(f"**Probability of Fake Job**: {prob:.2f}")

        if prob >= 0.5:
            st.error("⚠️ This job posting is likely **FAKE**.")
        else:
            st.success("✅ This job posting is likely **LEGIT**.")

