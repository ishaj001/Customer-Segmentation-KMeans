
# Customer Segmentation Using K-Means Clustering

This project aims to segment customers of a mall based on their annual income and spending score using K-Means clustering. The goal is to identify distinct customer groups that can be targeted for personalized marketing strategies.

## Dataset

The dataset used for this project is the *Mall_Customers.csv*, which contains information about customers including:
- **CustomerID**: Unique identifier for each customer
- **Gender**: Gender of the customer
- **Age**: Age of the customer
- **Annual Income**: Annual income of the customer
- **Spending Score**: Spending score assigned by the mall (based on customer behavior)

## Steps Involved:

1. **Loading and Cleaning Data**: The data is loaded into a Pandas DataFrame and cleaned for missing values.
2. **Data Preprocessing**: Selecting relevant features (Annual Income and Spending Score) for clustering.
3. **Elbow Method for Optimal Clusters**: Using the Elbow Method to determine the optimal number of clusters.
4. **K-Means Clustering**: Applying the K-Means algorithm to segment customers into clusters.
5. **Visualization**: Visualizing the clusters and their centroids on a 2D plot.

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`

## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Customer-Segmentation-KMeans.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place your `Mall_Customers.csv` file in the project directory.

4. Run the Python script:
   ```
   python customer_segmentation.py
   ```

## Results

The model segments the customers into 5 distinct groups based on their annual income and spending score. The results are visualized on a scatter plot with the clusters and their centroids.

## Example Output:

Here’s the plot you can expect after running the K-Means model:

- **Cluster 1** (Green): Customers with low annual income and low spending score.
- **Cluster 2** (Red): Customers with moderate annual income and low spending score.
- **Cluster 3** (Yellow): Customers with moderate annual income and high spending score.
- **Cluster 4** (Violet): Customers with high annual income and low spending score.
- **Cluster 5** (Blue): Customers with high annual income and high spending score.

