import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn
import math

#read dataset
dataset = pandas.read_csv('C:/Users/HP ENVY/PycharmProjects/UM/Weather.csv', low_memory=False)
print(dataset.shape)

print(dataset.describe())

#dataset.plot(x='MinTemp', y='MaxTemp', style='o')
#plt.title("Plot 1")
#plt.show() - all plots

X = dataset['MinTemp'].values.reshape(-1, 1)
Y = dataset['MaxTemp'].values.reshape(-1, 1)
#print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print(regressor.coef_)
print(regressor.intercept_)
print('R2 train', regressor.score(X_train, Y_train))

print('R2 test', regressor.score(X_test, Y_test))

Y_pred = regressor.predict(X_test)
df = pandas.DataFrame({'actual': Y_test.flatten(), 'pred': Y_pred.flatten()})
print(df)

#df.head(30).plot(kind='bar')
#plt.show()

plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred, color='red')
plt.show()

mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
mae = sklearn.metrics.mean_absolute_error(Y_test, Y_pred)
rmse = math.sqrt(mse)

print('MSE = ', mse)
print('MAE = ', mae)
print('RMSE = ', rmse)
