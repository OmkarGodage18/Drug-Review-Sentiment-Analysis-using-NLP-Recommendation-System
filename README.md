# Drug-Review-Sentiment-Analysis-using-NLP-Recommendation-System
AI-powered sentiment analysis and drug recommendation system built with Python, NLP, Flask API, and Scikit-learn.
# Drug Review Sentiment Analysis & Recommendation System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Flask](https://img.shields.io/badge/Flask-API-black?logo=flask)
![SQLite](https://img.shields.io/badge/SQLite-Database-blue?logo=sqlite)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green)

---

##  Project Overview

This project applies Natural Language Processing (NLP) and Machine Learning to analyze patient drug reviews, classify their sentiment, and recommend effective medications based on user reviews, ratings, and usefulness scores.

The system supports both batch processing and real-time predictions through a Flask API.

---

##  Objective

- Perform sentiment analysis on patient drug reviews.
- Recommend suitable drugs for specific medical conditions.
- Build an end-to-end NLP pipeline.
- Deploy the model using Flask API.

---

##  Dataset

**Dataset:** `drug_reviews_sentiment_analysis`

| Dataset | Records |
|---------|---------:|
| Training | 112,908 |
| Testing | 32,259 |
| Production | 16,130 |
| **Total** | **161,297** |

> The dataset is not included because of size limitations.

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- SQLite
- NLTK
- TF-IDF
- Pickle

---

##  Machine Learning Pipeline

### Data Preprocessing

- Text Cleaning
- Stopword Removal
- Lemmatization
- TF-IDF Vectorization

### Model Training

- Sentiment Classification
- Drug Recommendation
- Feature Engineering
- Model Evaluation

### Deployment

- Flask REST API
- Batch Prediction
- Model Serialization

---

##  Project Structure

```text
Drug-Review-Sentiment-Analysis
│
├── data
├── model
├── train.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
└── screenshots
```

---

##  Model Evaluation

| Metric | Value |
|---------|-------|
| Accuracy | 0.808 |
| Precision | 0.8112420866863809 |

---

## 📈 Features Used

- Drug Name
- Medical Condition
- User Review
- Rating
- Useful Count

---

##  Run the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train.py
```

### Predict

```bash
python predict.py
```

### Run Flask API

```bash
python app.py
```

---

##  Future Improvements

- BERT-based Sentiment Analysis
- Hugging Face Transformers
- Explainable AI (SHAP/LIME)
- Docker Deployment
- Azure/AWS Deployment
- LLM-powered Drug Recommendation

---

##  Author

**Omkar Godage**

AI & Machine Learning Engineer

GitHub: https://github.com/YourUsername



---
 If you found this project useful, please consider giving it a star.
