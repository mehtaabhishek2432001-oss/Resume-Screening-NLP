import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pypdf import PdfReader

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Same cleaning as the notebook
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# Read text from a PDF file
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + " "
    return text

# Load model once
@st.cache_resource
def load_model():
    return joblib.load("resume_classifier.pkl")

model = load_model()

# Predict top 3 categories
def predict(resume_text):
    data = pd.DataFrame({
        "clean": [clean_text(resume_text)],
        "title": [clean_text(" ".join(resume_text.split()[:10]))]
    })
    scores = model.decision_function(data)[0]
    top3 = np.argsort(-scores)[:3]
    return [model.classes_[i] for i in top3]

# ---------- Page ----------
st.set_page_config(page_title="Resume Screening AI", page_icon="📄")
st.title("📄 Resume Screening & Classification")
st.write("Upload a resume PDF or paste the text. The model will predict the job category.")

tab1, tab2 = st.tabs(["📎 Upload PDF", "📝 Paste Text"])

resume_text = ""

with tab1:
    pdf_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])
    if pdf_file is not None:
        resume_text = read_pdf(pdf_file)
        if resume_text.strip() == "":
            st.error("Could not read text from this PDF (it may be a scanned image).")
        else:
            with st.expander("See extracted text"):
                st.write(resume_text[:2000])

with tab2:
    pasted = st.text_area("Resume text", height=300,
                          placeholder="Example: SOFTWARE DEVELOPER  Summary  Python developer with...")
    if pasted.strip():
        resume_text = pasted

if st.button("Predict Category"):
    if len(resume_text.split()) < 10:
        st.warning("Please upload a PDF or paste a longer resume (at least 10 words).")
    else:
        top3 = predict(resume_text)
        st.success(f"Predicted Category: **{top3[0]}**")
        st.subheader("Top 3 Matches")
        for rank, cat in enumerate(top3, start=1):
            st.write(f"{rank}. {cat}")

st.caption("Model: Linear SVM + TF-IDF (resume body + job title) | Accuracy ~83% | Top-3 ~93%")
