from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

data = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1.8)
print(data)
plt.scatter(data[0][:,0],data[0][:,1], c=data[1])
plt.show()

from sklearn.cluster import KMeans
wcss=[]
for i in range(1, 20):
        kmeans=KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
        kmeans.fit(data[0])
        wcss.append(kmeans.inertia_)
plt.plot(range(1,20),wcss )
plt.show()

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10)
pred_y = kmeans.fit_predict(data[0])
plt.scatter(data[0][:,0], data[0][:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()

from pyclustering.cluster.kmedoids import kmedoids
initial_medoids = [100,200,300,400]
k_medoids = kmedoids (nclusters=4, data=data[0],initial_index_medoids=initial_medoids)
k_medoids.process()
pred_y = k_medoids.predict(data[0])

from pyclustering.cluster import cluster_visualizer
clusters = k_medoids.get_clusters()  # list of clusters
medoids = k_medoids.get_medoids()  # list of cluster centers.
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, data[0])
visualizer.append_cluster(medoids, data[0], markersize=20)
visualizer.show()


