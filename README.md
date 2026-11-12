# Resume Screening & Classification using NLP

**Innomatics Research Labs – NLP Capstone Project**
**Author:** Abhishek A Mehta

An NLP system that reads a resume and predicts which of **24 job categories** it belongs to (IT, HR, Finance, Chef, Teacher, …). It also includes a **Streamlit web app**: upload a resume PDF and get the predicted category plus the top 3 matches.

---

## 1. Problem Statement

Recruiters receive hundreds of resumes for every opening, and sorting them by hand is slow and inconsistent. This project automatically classifies each resume into the right job category so it can be routed to the right recruiter.

- **Task:** multi-class text classification (24 classes)
- **Metrics:** Accuracy, Macro F1 (treats small categories fairly) and Top-3 Accuracy

## 2. Dataset

- **Source:** Kaggle *Resume Dataset* (resumes from LiveCareer)
- **File:** `Resume.csv` with 2,484 resumes and 4 columns: `ID`, `Resume_str` (text), `Resume_html`, `Category` (target)
- **Categories:** 24, mostly 96–120 resumes each; Agriculture (63), Automobile (36) and BPO (22) are small
- **Cleaning:** 2 duplicates and 1 empty resume removed → 2,481 resumes
- **Split:** 80/20 stratified → 1,984 train, 497 test

## 3. Approach

| Stage | What was done |
|---|---|
| EDA | Category distribution, resume length, top words, word clouds |
| Preprocessing | Lowercase, keep letters only, tokenize, remove NLTK stopwords, WordNet lemmatization |
| Features | Bag of Words, TF-IDF, word embeddings (BiLSTM) |
| ML models | Multinomial Naive Bayes, Logistic Regression, Linear SVM (each with BoW and TF-IDF) |
| Advanced model | BiLSTM (Embedding 64 → BiLSTM 64 → Dropout 0.5 → Dense 64 → Softmax 24) |
| Improvement | Separate TF-IDF for the job title (first 10 words) + word-pair TF-IDF on the body, class-balanced Linear SVM, GridSearchCV (5-fold) |
| Evaluation | Accuracy, Macro F1, Top-3 accuracy, classification report, confusion matrix, error analysis |

## 4. Results (test set, 497 resumes)

| Model | Features | Accuracy | Macro F1 |
|---|---|---|---|
| **Improved Linear SVM (final)** | **TF-IDF body + job title** | **≈ 82.9%** | **≈ 0.78** |
| Linear SVM | TF-IDF | 68.6% | 0.644 |
| Logistic Regression | BoW | 65.8% | 0.628 |
| Linear SVM | BoW | 65.0% | 0.612 |
| Logistic Regression | TF-IDF | 65.0% | 0.589 |
| BiLSTM | Word embeddings | 63.8% | 0.582 |
| Naive Bayes | BoW | 56.3% | 0.519 |
| Naive Bayes | TF-IDF | 54.5% | 0.468 |

- **Top-3 accuracy of the final model: ≈ 93%.** The correct category is in the model's 3 best guesses for about 9 in 10 resumes.
- The job-title feature gave the biggest gain (+14 points). Hyperparameter tuning added less than 1 point.
- Many remaining errors are **mislabelled resumes** in the dataset (e.g. a "Kindergarten Teacher" labelled CHEF), so about 83% is close to the honest ceiling for this data.

Exact numbers can vary by about ±1% between machines and library versions.

## 5. Project Files

```
Resume_Screening-NLP/
├── Resume.csv                     # dataset
├── Resume_Screening_NLP.ipynb     # complete notebook (all 10 steps)
├── app.py                         # Streamlit web app (PDF upload + paste text)
├── resume_classifier.pkl          # final model (TF-IDF + Linear SVM pipeline)
├── bilstm_model.keras             # BiLSTM model (created when the notebook runs)
├── requirements.txt               # Python packages
├── Project_Report.pdf             # project report
└── README.md                      # this file
```

The presentation is a separate slide deck (exported as PPTX / PDF).

## 6. How to Run

### Step 1 – Set up the environment (Windows, Command Prompt)

```cmd
cd path\to\Resume_Screening-NLP
python -m venv myenv
myenv\Scripts\activate.bat
pip install -r requirements.txt
```

TensorFlow needs Python 3.9–3.12.

### Step 2 – Run the notebook

1. Open the folder in VS Code.
2. Open `Resume_Screening_NLP.ipynb` and choose the **myenv** kernel.
3. Click **Run All**. It takes about 10 minutes on a CPU, mostly for the BiLSTM.

The notebook saves `resume_classifier.pkl` and `bilstm_model.keras`.

### Step 3 – Run the web app

```cmd
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Upload a resume PDF (for example, any file from the dataset's `data/data/CHEF` folder) or paste resume text, then click **Predict Category**.

## 7. Using the Model in Python

```python
import joblib, pandas as pd, numpy as np
model = joblib.load("resume_classifier.pkl")
data = pd.DataFrame({"clean": [clean_text(text)],
                     "title": [clean_text(" ".join(text.split()[:10]))]})
print(model.predict(data)[0])
```

`clean_text` is the same cleaning function used in the notebook and in `app.py`.

## 8. Limitations & Future Scope

**Limitations:** noisy labels from keyword scraping, very small classes (BPO, Automobile), English/US resumes only, and scanned PDFs need OCR.

**Future scope:** fine-tune BERT/DistilBERT, clean or relabel noisy resumes, match resumes to job descriptions, extract skills and experience, and deploy on Streamlit Cloud.

## 9. Tech Stack

Python · pandas · NumPy · NLTK · scikit-learn · TensorFlow/Keras · Matplotlib · Seaborn · WordCloud · Streamlit · pypdf · joblib
