from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np

data = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1.8)
X=data[0]
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)
plt.scatter(X[:,0],X[:,1], c=data[1])
plt.show()
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.25, min_samples=5)
y_pred = dbscan.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Paired')
plt.title("DBSCAN")
plt.show()

from sklearn.cluster import OPTICS
clust = OPTICS(min_samples=50)
clust.fit(X)
space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    plt.plot(Xk, Rk, color, alpha=0.3)
plt.show()

