from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

X, y = make_blobs(n_samples=100, centers=4)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X = np.dot(X, transformation)
plt.scatter(X[:,0],X[:,1], c=y)
plt.show()

from sklearn.mixture import GaussianMixture
gm= GaussianMixture(n_components=4, covariance_type='spherical')
gm.fit(X)
print(gm.means_)
print(gm.covariances_)

X_, Y = np.meshgrid(np.linspace(-20, 20), np.linspace(-20, 20))
XX = np.array([X_.ravel(), Y.ravel()]).T
Z = -gm.score_samples(XX)
Z = Z.reshape((50, 50))
plt.contour(X_, Y, Z,norm=LogNorm(vmin=0.001, vmax=1000.0),levels=np.logspace(0, 3, 10))
plt.scatter(X[:, 0], X[:, 1])
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(init='k-means++', max_iter=300, n_init=10)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()
