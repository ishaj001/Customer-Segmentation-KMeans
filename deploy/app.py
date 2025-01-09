import joblib
import numpy as np
from flask import Flask, render_template, request
import os
from pathlib import Path

app = Flask(__name__)

# Use relative path for model
model_path = Path(__file__).parent / 'models' / 'kmeans_model.pkl'
model = joblib.load(model_path)

# Define cluster descriptions
cluster_descriptions = {
    0: "Cluster 1: Moderate Income, Moderate Spending",
    1: "Cluster 2: High Income, High Spending",
    2: "Cluster 3: High Income, Low Spending",
    3: "Cluster 4: Low Income, Low Spending",
    4: "Cluster 5: Low Income, High Spending"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    result_message = "Prediction will be shown here after clicking 'Predict'"
    cluster_description = ""
    
    if request.method == 'POST':
        # Retrieve input data from the form and convert them to floats
        annual_income = float(request.form['annual_income'])
        spending_score = float(request.form['spending_score'])
        
        # Prepare input data for prediction
        input_data = np.array([[annual_income, spending_score]])

        # Make prediction with KMeans model
        predicted_cluster = model.predict(input_data)[0]

        # Get the description of the predicted cluster
        cluster_description = cluster_descriptions.get(predicted_cluster, "No description available for this cluster.")
        
        # Return the result based on prediction
        result_message = f"Predicted Cluster: {predicted_cluster}"

    return render_template("index.html", prediction=result_message, cluster_description=cluster_description)

if __name__ == '__main__':
    # Change the host to '0.0.0.0' to make the app accessible externally
    app.run(host='0.0.0.0', port=5000, debug=True)
