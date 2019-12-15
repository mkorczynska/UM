import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

pd.options.display.max_columns = 999
data = pd.read_csv('train.csv')

g0 = sns.FacetGrid(data, col='Survived')
g0.map(plt.hist, 'Age', bins=20)
g1 = sns.FacetGrid(data, col='Survived')
g1.map(plt.hist, 'Fare', bins=20)
g2 = sns.FacetGrid(data, col='Survived', row='Pclass')
g2.map(plt.hist, 'Age', bins=20)
g3 = sns.FacetGrid(data, col='Survived')
g3.map(plt.hist, 'Age', bins=20)
g4 = sns.FacetGrid(data, col='Survived', row='Embarked')
g4.map(plt.hist, 'Age', bins=20)
plt.show()

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
# -----------------------------------------
data = pd.read_csv('train.csv')
labelEncoder = LabelEncoder()
labelEncoder.fit(data['Sex'])
data['Sex'] = labelEncoder.transform(data['Sex'])
y = data['Survived']
data = data.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'Survived'], axis=1)
data = data.fillna(data.mean())
X = data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print('Original number of features:', X_train.shape[1])
print('Reduced number of features:', X_train_pca.shape[1])
