import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import re

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA as sklearnPCA

pd.options.display.max_columns = 999
data = pd.read_csv('train.csv')

# g0 = sns.FacetGrid(data, col='Survived')
# g0.map(plt.hist, 'Age', bins=20)
# g1 = sns.FacetGrid(data, col='Survived')
# g1.map(plt.hist, 'Fare', bins=20)
# g2 = sns.FacetGrid(data, col='Survived', row='Pclass')
# g2.map(plt.hist, 'Age', bins=20)
# g3 = sns.FacetGrid(data, col='Survived')
# g3.map(plt.hist, 'Age', bins=20)
# g4 = sns.FacetGrid(data, col='Survived', row='Embarked')
# g4.map(plt.hist, 'Age', bins=20)
# plt.show()

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
print("Correctly classified: ", correct)
print("Correctly classified %: ", correct / len(X))

data['Survived'] = y
data = data.drop(['PassengerId'], axis=1)
print(data.corr())
# -----------------------------------------

