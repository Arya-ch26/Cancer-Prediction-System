from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and preprocessing pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get data from HTML form
    data = {
        "Age": float(request.form["Age"]),
        "Gender": int(request.form["Gender"]),
        "BMI": float(request.form["BMI"]),
        "Smoking": int(request.form["Smoking"]),
        "GeneticRisk": int(request.form["GeneticRisk"]),
        "PhysicalActivity": float(request.form["PhysicalActivity"]),
        "AlcoholIntake": float(request.form["AlcoholIntake"]),
        "CancerHistory": int(request.form["CancerHistory"])
    }

    # Convert input into DataFrame
    input_df = pd.DataFrame([data])

    # Apply preprocessing
    input_prepared = pipeline.transform(input_df)

    # Make prediction
    prediction = model.predict(input_prepared)[0]

    # Prediction confidence (only for models that support predict_proba)
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_prepared)[0][prediction]

    return render_template(
        "result.html",
        result=prediction,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)