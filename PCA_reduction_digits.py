from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

digits = datasets.load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
pca = PCA(n_components=0.8)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print('Original number of features:', X_train.shape[1])
print('Reduced number of features:', X_train_pca.shape[1])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logisticRegr = LogisticRegression(solver = 'lbfgs',multi_class='auto',max_iter=1000)
logisticRegr.fit(X_train_pca, y_train)
predicted=logisticRegr.predict(X_test_pca)
print ('Accuracy LR:', accuracy_score(y_test, predicted))