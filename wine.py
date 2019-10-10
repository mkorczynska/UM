import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn
import math

dataset = pandas.read_csv('C:/Users/HP ENVY/PycharmProjects/UM/winequality.csv', low_memory=False)
print(dataset.shape)

print(dataset.describe())

dataset = dataset.fillna(method='ffill')

X = dataset[{'pH', 'sulphates', 'alcohol', 'chlorides'}].values
Y = dataset['quality'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print(regressor.coef_)
print(regressor.intercept_)
print(regressor.score(X_test, Y_test))

Y_pred = regressor.predict(X_test)

mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
mae = sklearn.metrics.mean_absolute_error(Y_test, Y_pred)
rmse = math.sqrt(mse)

print('MSE = ', mse)
print('MAE = ', mae)
print('RMSE = ', rmse)
