from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

# Create FastAPI instance
app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing) to allow the front-end to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify specific domains instead of "*" (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model (Ensure this is the correct path to your saved model)
model_path = 'C:/Users/intern_datascience/Documents/deploy/models/kmeans_model.pkl'
model = joblib.load(model_path)

# Create the input data model
class CustomerData(BaseModel):
    annual_income: float
    spending_score: float

# Define the clusters and descriptions
cluster_labels = {
    0: 'Cluster 1',
    1: 'Cluster 2',
    2: 'Cluster 3',
    3: 'Cluster 4',
    4: 'Cluster 5',
}

cluster_descriptions = {
    0: 'Low income, Low spending',
    1: 'High income, High spending',
    2: 'High income, Low spending',
    3: 'Low income, High spending',
    4: 'Medium income, Medium spending',
}

# Root route
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return HTMLResponse(content=f.read())

# Define the prediction route
@app.post("/predict")
async def predict(data: CustomerData):
    # Prepare the data for prediction
    X_new = np.array([[data.annual_income, data.spending_score]])

    # Make the prediction
    predicted_cluster = model.predict(X_new)[0]

    # Get the cluster name and description
    cluster_label = cluster_labels.get(predicted_cluster, "Unknown Cluster")
    cluster_description = cluster_descriptions.get(predicted_cluster, "No description available")

    # Return the prediction result with both cluster ID and description
    return {
        "predicted_cluster": cluster_label,
        "cluster_description": cluster_description,
        "annual_income_unit": "k$"  # Indicating that the income is in thousands of dollars
    }

# Serve static files (like CSS) from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")
