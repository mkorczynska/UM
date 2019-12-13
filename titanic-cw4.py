import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as sklearnPCA

pd.options.display.max_columns = 999
data = pd.read_csv('train.csv')
# print("Data\n", data)
print("Data head\n", data.head())
print("Data describe\n", data.describe())

labelEncoder = LabelEncoder()
labelEncoder.fit(data['Sex'])
data['Sex'] = labelEncoder.transform(data['Sex'])
y = data['Survived']
data = data.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'Survived'], axis=1)
data = data.fillna(data.mean())
print("Data after changes: \n", data.describe())

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
correct = 0
X = np.array(data).astype(float)
for i in range(len(X)):
    pred = np.array(X[i].astype(float))
    pred = pred.reshape(-1, len(pred))
    pred = kmeans.predict(pred)
    if pred == y[i]:
        correct += 1
print(correct)
print(correct / len(X))

labels = kmeans.labels_
print(labels)
# check how many of the samples were correctly labeled
# correct_labels = sum(y_var == labels)
#
# print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y_var.size))
#
# sklearn_pca = sklearnPCA(n_components=3)
# Y_sklearn = sklearn_pca.fit_transform(X_std)
# sklearn_pca.explained_variance_ratio_
# Y_sklearn.shape
#
# sklearn_pca.transform()
