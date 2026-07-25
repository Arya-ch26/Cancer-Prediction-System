import os
from xml.parsers.expat import model
from flask import request
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


def build_pipeline(num_attribs):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler()),
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
    ])
    
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    cancer = pd.read_csv("The_cancer_data_1500_V2.csv")  

    cancer["Age_cat"] = pd.cut(
        cancer["Age"],
        bins=[18, 30, 40, 50, 60, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(cancer, cancer["Age_cat"]):
        cancer.loc[test_index].drop("Age_cat", axis=1).to_csv("input.csv", index=False)
        

        cancer = cancer.loc[train_index].drop("Age_cat", axis=1)
         

    cancer_labels = cancer["Diagnosis"].copy()
    cancer = cancer.drop("Diagnosis", axis=1)    
        
    num_attribs = cancer.columns.tolist()
    
    pipeline = build_pipeline(num_attribs)
    cancer_prepared = pipeline.fit_transform(cancer)
    

    model = RandomForestClassifier(random_state=42)
    model.fit(cancer_prepared, cancer_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model and pipeline saved successfully.")

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    input_prepared = pipeline.transform(input_data)
    predictions = model.predict(input_prepared)
    input_data["Predicted_Diagnosis"] = predictions
    input_data.to_csv("output.csv", index=False)
    print("Predictions saved to output.csv.")
