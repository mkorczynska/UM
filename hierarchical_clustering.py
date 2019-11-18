import numpy as np
import matplotlib.pyplot as plt

X = np.array([[5,3],
    [10,15],
    [15,12],
    [24,10],
    [30,30],
    [85,70],
    [71,80],
    [60,78],
    [70,55],
    [80,91],])

labels = range(1, 11)
plt.scatter(X[:,0],X[:,1])

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-5, 0),
        textcoords='offset points', ha='right', va='center')
plt.show()

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(affinity='euclidean', linkage='ward', distance_threshold=50, n_clusters=None)
cluster.fit_predict(X)
print(cluster.labels_)
plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(X, 'single')
labelList = range(1, 11)

dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending')
plt.show()

import scipy.cluster.hierarchy as shc
dend = shc.dendrogram(shc.linkage(X, method='ward'))
plt.show()