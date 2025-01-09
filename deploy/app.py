import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import os
from pathlib import Path

app = Flask(__name__)


# Use relative path for model
model_path = Path(__file__).parent / 'models' / 'heart_pred_model.pkl'
model = joblib.load(model_path)


# Define the column names that the model was trained on
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@app.route('/', methods=['GET', 'POST'])
def index():
    result_message = "Prediction will be shown here after clicking 'Predict'"
    
    if request.method == 'POST':
        # Retrieve input data from the form and convert them to floats
        input_data = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal']),
        ]
        
        # Convert the input data to a pandas DataFrame with the correct column names
        input_data_as_df = pd.DataFrame([input_data], columns=columns)

        # Make prediction with DataFrame
        prediction = model.predict(input_data_as_df)

        # Return the result based on prediction
        if prediction[0] == 0:
            result_message = "The Person does not have Heart Disease"
        else:
            result_message = "The Person has Heart Disease"
    
    return render_template("index.html", prediction=result_message)

if __name__ == '__main__':
    app.run(debug=True)
