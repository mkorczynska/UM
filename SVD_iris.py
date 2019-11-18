from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
X = StandardScaler().fit_transform(X)

from numpy import *
r = 2
U, s, V = linalg.svd(X)
Sig = mat(eye(r)*s[:r])
X_r = U[:,:r]
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
   plt.scatter(X_r[y==i,0], X_r[y==i,1], c=c, label=target_name)
plt.legend()
plt.title('SVD of IRIS dataset')
plt.show()
