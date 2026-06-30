#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    f1_score
)


def fit_and_evaluate_model(
        x_train,
        x_val,
        y_train,
        y_val
):

    model = LogisticRegression(
        max_iter=2000,
        solver='saga',
        class_weight='balanced',
        C=2,
        n_jobs=-1,
        random_state=42
    )

    # Train
    model.fit(x_train, y_train)

    # Predict
    pred = model.predict(x_val)

    # Evaluation
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_val, pred))

    print("\nAccuracy:")
    print(accuracy_score(y_val, pred))

    print("\nF1 Score:")
    print(f1_score(y_val, pred, average='weighted'))

    print("\nClassification Report:\n")
    print(classification_report(y_val, pred))

    return model


def get_feature_importance(model, tfidf):

    feature_names = tfidf.get_feature_names_out()

    coefficients = model.coef_[0]

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': coefficients
    })

    importance_df = importance_df.sort_values(
        by='importance',
        ascending=False
    )

    return importance_df


# In[2]:


#get_ipython().system('jupyter nbconvert --to script model_building.ipynb')


# In[ ]:




