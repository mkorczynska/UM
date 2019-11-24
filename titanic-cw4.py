import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pd.options.display.max_columns = 999
data = pd.read_csv('train.csv')
print(data)
print(data.head())
print(data.describe())
print(data.info())

# df = pd.DataFrame(data)
# cor = df.corr()
# sns.heatmap(cor, annot=True)
# plt.show()

from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()
labelEncoder.fit(data['Sex'])
data['Sex'] = labelEncoder.transform(data['Sex'])
y = data['Survived']
data = data.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'Survived'], axis=1)
data = data.fillna(data.mean())
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
correct = 0
X = np.array(data).astype(float)
for i in range(len(X)):
    pred = np.array(X[i].astype(float))
    pred = pred.reshape(-1, len(pred))
    pred = kmeans.predict(pred)
    if (pred == y[i]):
        correct += 1
print(correct / len(X))
