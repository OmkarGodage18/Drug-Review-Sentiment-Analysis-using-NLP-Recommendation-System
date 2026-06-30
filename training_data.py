

import pandas as pd
import joblib
import os

from data_preprocessing import (
    text_data_cleaning,
    tfidf_feature_fit,
    tfidf_features_transform,
)

from data_splitting import prepare_data

from model_building import (
    fit_and_evaluate_model
)





data = pd.read_csv('drug_reviews_sentiment_analysis.csv')


train_data = data.iloc[0:112908]
val_data   = data.iloc[112908:145167]
live_data  = data.iloc[145167:]

print("train data shape:", train_data.shape)
print("val data shape:", val_data.shape)
print("live data shape:", live_data.shape)

train_data = train_data.head(5000)
val_data   = val_data.head(2000)
live_data  = live_data.head(2000)

train_data = text_data_cleaning(train_data)
val_data   = text_data_cleaning(val_data)
live_data  = text_data_cleaning(live_data)


tfidf, x_train = tfidf_feature_fit(train_data)
x_val  = tfidf_features_transform(tfidf, val_data)
x_live = tfidf_features_transform(tfidf, live_data)




y_train = train_data['sentiment']
y_val   = val_data['sentiment']



model = fit_and_evaluate_model(x_train, x_val, y_train, y_val)



joblib.dump(model, "model_classifier.pkl")
joblib.dump(tfidf, "tfidf.pkl")

print("Model and TF-IDF saved successfully ")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

y_pred = model.predict(x_val)

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(7,5))
plt.bar(metrics, values)

for i, v in enumerate(values):
    plt.text(i, v+0.01, f"{v:.3f}", ha="center")

plt.ylim(0,1.1)
plt.title("Model Performance")
plt.savefig("Accuracy.png", dpi=300)
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(
    model,
    x_val,
    y_val,
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.savefig("Prediction.png", dpi=300)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

feature_names = tfidf.get_feature_names_out()

importance = np.abs(model.coef_[0])

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
).head(20)

plt.figure(figsize=(10,6))
plt.barh(feature_df["Feature"], feature_df["Importance"])
plt.gca().invert_yaxis()

plt.title("Top 20 Important Words")
plt.xlabel("Coefficient Magnitude")

plt.tight_layout()
plt.savefig("FeatureImportance.png", dpi=300)
plt.show()