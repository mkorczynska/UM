import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn
import math
from sklearn.datasets import load_boston

dataset = load_boston()
print(dataset.keys())
print(dataset.feature_names)

df = pandas.DataFrame(dataset.data)
print(df.head())

df.columns = dataset.feature_names
print(df.head())

df['PRICE'] = dataset.target
print(df.describe())

X = df.drop('PRICE', axis=1)
Y = df['PRICE']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print(regressor.coef_)
print(regressor.intercept_)
print('R2 train', regressor.score(X_train, Y_train))
print('R2 test', regressor.score(X_test, Y_test))
