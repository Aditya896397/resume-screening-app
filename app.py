import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import pdfplumber
from docx import Document
import io
from fuzzywuzzy import fuzz
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Resume Screener Pro", layout="wide")

# ---------------------------
# Utilities: parsing & features
# ---------------------------

SKILL_VOCAB = [
    "python", "sql", "aws", "docker", "kubernetes", "pandas", "numpy", "git", "ci/cd",
    "tensorflow", "pytorch", "scikit-learn", "nlp", "nlp", "spark", "airflow", "etl",
    "react", "node", "flask", "fastapi", "linux", "bash", "rest", "api", "tableau",
    "powerbi", "excel", "hadoop", "nosql", "mongodb", "postgresql", "mysql"
]

def extract_text_from_pdf(file_bytes):
    text = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text.append(t)
    except Exception as e:
        st.warning("PDF parser error: " + str(e))
    return "\n".join(text)

def extract_text_from_docx(file_bytes):
    try:
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        st.warning("DOCX parser error: " + str(e))
        return ""

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(\+?\d{1,3}[-.\s]?)?(\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})')

def extract_contact_info(text):
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    # PHONE_RE returns tuples due to groups — join and clean
    phones_clean = []
    for p in phones:
        joined = "".join(p)
        # keep only digits and leading +
        joined = re.sub(r'[^\d+]', '', joined)
        if len(joined) >= 7:
            phones_clean.append(joined)
    return list(dict.fromkeys(emails)), list(dict.fromkeys(phones_clean))

def extract_years_experience(text):
    # heuristics: look for "3 years", "5+ years", "X yrs", or date ranges like 2018-2022 -> compute approx
    # 1) simple patterns
    m = re.findall(r'(\d+(?:\.\d+)?)[\+\s]*(?:years|yrs|year)', text, flags=re.I)
    if m:
        # return the maximum number found (assume total experience mentions)
        try:
            return float(max(m, key=lambda x: float(x)))
        except:
            pass
    # 2) date ranges -> try estimate
    years = re.findall(r'(\b(19|20)\d{2})\b', text)
    years = [int(y[0]) for y in years]
    if len(years) >= 2:
        return float(max(years) - min(years))
    return 0.0

def extract_education_level(text):
    t = text.lower()
    if "ph.d" in t or "phd" in t or "doctor" in t:
        return "PhD"
    if "master" in t or "m.s" in t or "msc" in t or re.search(r'\bm\.\s?', t):
        return "Master"
    if "bachelor" in t or "b\.s" in t or "bsc" in t or re.search(r'\bb\.\s?', t):
        return "Bachelor"
    return "Unknown"

def extract_skills(text, vocab=SKILL_VOCAB, fuzz_threshold=85):
    found = []
    low = text.lower()
    # exact word presence check
    for skill in vocab:
        if skill.lower() in low:
            found.append(skill.lower())
    # fuzzy phrase matching for vocab not caught
    for skill in vocab:
        if skill.lower() in found:
            continue
        # try fuzzy with sliding windows of text tokens (costly for very long text)
        score = fuzz.partial_ratio(skill.lower(), low)
        if score >= fuzz_threshold:
            found.append(skill.lower())
    # also surface top fuzzy matches from vocab (useful for short resumes)
    # return unique
    return sorted(list(set(found)))

def highlight_keywords(text, keywords):
    # naive highlight by wrapping matched keywords in **bold** markdown
    out = text
    for kw in sorted(set(keywords), key=lambda x: -len(x)):
        # word boundary replace, case-insensitive
        pattern = re.compile(re.escape(kw), flags=re.I)
        out = pattern.sub(f"**{kw}**", out)
    return out

# ---------------------------
# Embeddings & similarity
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def cosine_sim(a, b):
    # expects 1D numpy arrays (or 2D with shape (1, dim))
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True)
    return float((a_n @ b_n.T).squeeze())

# ---------------------------
# UI / App layout
# ---------------------------
st.sidebar.title("Resume Screener Pro")
st.sidebar.markdown("Settings & controls")

model_name = st.sidebar.selectbox("Embedding model", ["sentence-transformers/all-MiniLM-L6-v2"], index=0)
threshold = st.sidebar.slider("Suitable threshold (similarity)", 0.0, 1.0, 0.70, 0.01)
fuzz_threshold = st.sidebar.slider("Skill fuzzy threshold", 70, 100, 88, 1)

# session storage for history
if "history" not in st.session_state:
    st.session_state.history = []

# Main layout
st.title("Resume Screener Pro — Enhanced MVP")
st.markdown(
    """
    Upload a resume (PDF/DOCX) or paste text. Paste a Job Description (JD).  
    The app extracts contact info, skills, approximate years, and computes semantic similarity using Sentence-BERT.
    """
)

col1, col2 = st.columns([1.2, 1])
with col1:
    st.subheader("Resume Input")
    upload = st.file_uploader("Upload resume (PDF or DOCX) — or skip and paste below", type=["pdf", "docx", "txt"])
    resume_text = st.text_area("Or paste resume text here", height=250)
    if upload:
        raw = upload.read()
        if upload.type == "application/pdf" or upload.name.lower().endswith(".pdf"):
            parsed_text = extract_text_from_pdf(raw)
        elif upload.name.lower().endswith(".docx"):
            parsed_text = extract_text_from_docx(raw)
        else:
            try:
                parsed_text = raw.decode("utf-8")
            except:
                parsed_text = ""
        # if pasted box empty, populate it
        if not resume_text.strip():
            resume_text = parsed_text
    # small quick example
    if st.button("Load Example Resume"):
        resume_text = (
            "John Doe\nEmail: john.doe@example.com\nPhone: +91 9876543210\n"
            "Experienced Data Engineer with 4 years experience in Python, SQL, AWS, Docker, Airflow. "
            "Built ETL pipelines and deployed services using Docker. B.Tech in Computer Science (2016-2020)."
        )

with col2:
    st.subheader("Job Description (JD)")
    jd_text = st.text_area("Paste JD here", height=350)
    if st.button("Load Example JD"):
        jd_text = (
            "Hiring: Python Data Engineer with 3+ years in ETL, SQL, AWS. "
            "Familiarity with Docker, Airflow, and data pipelines required."
        )

# Action
if st.button("Screen Candidate"):
    if not resume_text.strip():
        st.error("Please upload or paste a resume.")
    elif not jd_text.strip():
        st.error("Please paste a Job Description.")
    else:
        model = load_model(model_name)
        with st.spinner("Extracting features and computing similarity..."):
            # parsing
            email, phones = extract_contact_info(resume_text)
            years = extract_years_experience(resume_text)
            edu = extract_education_level(resume_text)
            skills = extract_skills(resume_text, SKILL_VOCAB, fuzz_threshold=fuzz_threshold)
            # embeddings
            emb_jd = model.encode([jd_text], convert_to_numpy=True)[0]
            emb_res = model.encode([resume_text], convert_to_numpy=True)[0]
            sim = cosine_sim(emb_jd, emb_res)
            label = "Not Suitable"
            if sim >= threshold:
                label = "Suitable"
            elif sim >= (threshold - 0.12):
                label = "Maybe / Trainable"

            # Save to history
            record = {
                "timestamp": datetime.now().isoformat(timespec='seconds'),
                "email": email[0] if email else "",
                "phones": ";".join(phones) if phones else "",
                "years_experience": years,
                "education": edu,
                "skills": ",".join(skills),
                "similarity": round(sim, 4),
                "label": label
            }
            st.session_state.history.insert(0, record)  # latest first

        # Results display
        st.success(f"Result: {label}")
        st.metric("Semantic Similarity", f"{sim:.3f}")

        st.markdown("**Extracted contact**")
        st.write({"email": email, "phones": phones})

        st.markdown("**Key extracted fields**")
        st.write({
            "years_experience": years,
            "education": edu,
            "skills": skills
        })

        st.markdown("**Matched skills / highlights (in resume preview)**")
        if skills:
            st.write(", ".join(skills))
            highlighted = highlight_keywords(resume_text, skills)
            st.markdown("----\n" + highlighted + "\n----")
        else:
            st.info("No skills from vocabulary found in resume text.")

        st.markdown("**Why this score?**")
        st.write("Similarity is semantic (sentence-transformer). Combine with skills & years for production-ready ranking.")

# History & download
st.markdown("---")
st.subheader("Screening History (current session)")
if st.session_state.history:
    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist, use_container_width=True)
    csv = df_hist.to_csv(index=False).encode('utf-8')
    st.download_button("Download history as CSV", data=csv, file_name="screening_history.csv", mime="text/csv")
    # similarity distribution plot
    st.markdown("**Similarity distribution (session)**")
    fig, ax = plt.subplots()
    ax.hist(df_hist["similarity"], bins=8)
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Similarity distribution")
    st.pyplot(fig)
else:
    st.info("No candidates screened in this session yet. Use 'Screen Candidate' to add records.")

# Footer / tips
st.sidebar.markdown("---")
st.sidebar.markdown("Tips:")
st.sidebar.markdown(
    """
- Tweak the 'Suitable threshold' slider based on your JD strictness.  
- Add or extend SKILL_VOCAB to match target roles.  
- Replace heuristics (years/education) with a proper parser (pyresparser / spaCy) for production.  
- Add SHAP or feature importance if you train a classifier later.
"""
)
