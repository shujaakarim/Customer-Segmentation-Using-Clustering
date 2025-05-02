# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Display first few rows
print(data.head())

# Basic info
print(data.info())

# Statistical summary
print(data.describe())

# ------------------------ Exploratory Data Analysis ------------------------

# Gender count plot
sns.countplot(x='Gender', data=data)
plt.title('Gender Distribution')
plt.show()

# Age distribution
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Income vs Spending Score scatter plot
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=data, hue='Gender')
plt.title('Income vs Spending Score')
plt.show()

# ------------------------ Feature Selection ------------------------

# Selecting relevant numerical features only (excluding 'Gender')
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Correlation heatmap
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# ------------------------ Data Preprocessing ------------------------

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------ Elbow Method to Choose K ------------------------

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot WCSS to find the elbow
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# ------------------------ Apply K-Means Clustering ------------------------

# From the elbow method, choose optimal k (e.g., 5)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to original data
data['Cluster'] = clusters

# ------------------------ Visualize Clusters ------------------------

# 2D plot: Annual Income vs Spending Score
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='Set1')
plt.title('Customer Segments')
plt.legend(title='Cluster')
plt.show()

# 3D visualization (optional)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Age'], data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='Set1')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score')
ax.set_title('3D View of Customer Segments')
plt.show()
