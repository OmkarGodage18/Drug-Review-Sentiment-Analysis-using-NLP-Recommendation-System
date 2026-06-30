#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

from data_preprocessing import (
    text_data_cleaning,
    tfidf_features_transform
)

# =========================================
# FLASK APP
# =========================================

app = Flask(__name__)

# =========================================
# LOAD MODEL & TFIDF
# =========================================

model = joblib.load("sentiment_model.pkl")

tfidf = joblib.load("tfidf_vectorizer.pkl")

# =========================================
# HOME PAGE
# =========================================

@app.route("/")
def home():

    return render_template("SENTIMENTs.html")

# =========================================
# WEB PREDICTION
# =========================================

@app.route('/predict', methods=["POST"])
def predict():

    review = request.form.get('Review')

    # Validation
    if not review or review.strip() == "":
        return render_template(
            "SENTIMENTs.html",
            prediction_text="Please enter a review"
        )

    # Create dataframe
    data_pred = pd.DataFrame(
        [review],
        columns=['review']
    )

    # Preprocess
    data_pred = text_data_cleaning(data_pred)

    # TFIDF transform
    data_pred_matrix = tfidf_features_transform(
        tfidf,
        data_pred
    )

    # Prediction
    prediction = model.predict(data_pred_matrix)[0]

    # Probability
    probability = model.predict_proba(data_pred_matrix).max()

    return render_template(
        "SENTIMENTs.html",
        prediction_text=f"Sentiment: {prediction}",
        probability_text=f"Confidence: {probability:.2%}"
    )

# =========================================
# API PREDICTION
# =========================================

@app.route('/sentiment', methods=["GET"])
def sentiment():

    review = request.args.get("Review")

    # Validation
    if not review or review.strip() == "":
        return jsonify({
            "error": "No review provided"
        })

    # Create dataframe
    data_pred = pd.DataFrame(
        [review],
        columns=['review']
    )

    # Preprocess
    data_pred = text_data_cleaning(data_pred)

    # TFIDF transform
    data_pred_matrix = tfidf_features_transform(
        tfidf,
        data_pred
    )

    # Prediction
    prediction = model.predict(data_pred_matrix)[0]

    # Probabilities
    probabilities = model.predict_proba(
        data_pred_matrix
    )[0]

    return jsonify({

        "review": review,

        "predicted_sentiment": str(prediction),

        "negative_probability": float(probabilities[0]),

        "positive_probability": float(probabilities[1])

    })

# =========================================
# LIVE CSV PREDICTION
# =========================================

@app.route('/predict_live', methods=["GET"])
def predict_live():

    # Load live data
    live_data = pd.read_csv("live_data.csv")

    # Fill null reviews
    live_data['review'] = live_data['review'].fillna('')

    # Take sample rows
    sample_data = live_data.head(10)

    # Preprocess
    sample_data = text_data_cleaning(sample_data)

    # TFIDF transform
    live_matrix = tfidf_features_transform(
        tfidf,
        sample_data
    )

    # Predictions
    predictions = model.predict(live_matrix)

    probabilities = model.predict_proba(live_matrix)

    results = []

    for i in range(len(sample_data)):

        results.append({

            "review":
            sample_data.iloc[i]['review_original'],

            "prediction":
            str(predictions[i]),

            "negative_probability":
            float(probabilities[i][0]),

            "positive_probability":
            float(probabilities[i][1])

        })

    return jsonify(results)

# =========================================
# MAIN
# =========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )


# In[ ]:


get_ipython().system('jupyter nbconvert --to script app.ipynb')


# In[ ]:




