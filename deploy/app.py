import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from pathlib import Path
from flask_cors import CORS  # Import CORS for handling cross-origin requests

app = Flask(__name__)

# Enable CORS
CORS(app)

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

# Function to handle prediction
def make_prediction(annual_income, spending_score):
    input_data = np.array([[annual_income, spending_score]])
    predicted_cluster = model.predict(input_data)[0]
    cluster_description = cluster_descriptions.get(predicted_cluster, "No description available for this cluster.")
    return predicted_cluster, cluster_description

@app.route('/')
def index():
    return render_template("index.html")  # Ensure your frontend template exists

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input from the request
        data = request.get_json()
        annual_income = float(data['annual_income'])
        spending_score = float(data['spending_score'])
        
        # Make prediction
        predicted_cluster, cluster_description = make_prediction(annual_income, spending_score)
        
        # Return prediction and cluster description as JSON
        return jsonify({
            "predicted_cluster": predicted_cluster,
            "cluster_description": cluster_description
        })
    except (ValueError, KeyError) as e:
        return jsonify({"error": "Invalid input. Please provide 'annual_income' and 'spending_score'."}), 400

if __name__ == '__main__':
    # Run on host 0.0.0.0 and port 8000 for external access
    app.run(host='0.0.0.0', port=8000, debug=True)
