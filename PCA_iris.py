from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
X = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)
print('explained variance ratio',pca.explained_variance_ratio_)
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
   plt.scatter(X_r[y==i,0], X_r[y==i,1], c=c, label=target_name)
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()

