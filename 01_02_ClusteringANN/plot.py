import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load your data from a file
data = pd.read_csv('unlabeledData2.txt', header=None)  # Ensure the path and header handling are correct
print(data)
# Choose the number of clusters
k = 5  # Modify this value based on your clustering needs

# Perform K-means clustering
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(data)

# Output the cluster labels
cluster_labels = kmeans.labels_

# Print the cluster labels for each sample
print("Cluster labels:")
print(cluster_labels)

# Optionally visualize the clusters if the data is reducible to 2 or 3 dimensions (this is optional and might not be feasible with high dimensions)
# Here's a simple visualization assuming the first two attributes represent meaningful dimensions
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.title('Cluster visualization based on the first two attributes')
plt.xlabel('Attribute 1')
plt.ylabel('Attribute 2')
plt.colorbar(label='Cluster Label')
plt.show()