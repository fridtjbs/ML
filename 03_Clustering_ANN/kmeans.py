import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('unlabeledData.txt', delimiter=' ')  # Replace with your file path

scores=[]

#print(data)
#print(data.max)
#print(data.min)
#print(data.mean)

#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

cluster_centers =[]

for i in range(2,16):

    Kmean = KMeans(n_clusters=i,init='k-means++', n_init=10, max_iter=300) #setting n_init 10 removes elbow
    print(i)

    

    Kmean.fit(data)
    cluster_centers.append(Kmean.cluster_centers_)

    scores.append(Kmean.inertia_)

print(scores)

#cluster_centers.append(Kmean.cluster_centers_)
#plt.scatter(cluster_centers[0][0],cluster_centers[0][1], color='blue', marker='o', label='Data points')
#plt.xlim(0, 1)  
#plt.ylim(0, 1) 
#plt.show()

plt.plot(range(2,16), scores)
plt.xlabel("K")
plt.ylabel("WCSS score")
plt.show()

