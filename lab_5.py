from keras import Sequential
from keras.layers import Dense
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn

# read dataset
dataset = pandas.read_csv('C:/Users/HP ENVY/Downloads/Weather.csv', low_memory=False)
print(dataset.shape)
print(dataset.describe())

X = dataset['MinTemp'].values.reshape(-1, 1)
Y = dataset['MaxTemp'].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# MLP
model = Sequential()
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
model.fit(X_train, Y_train, epochs=10)
print(model.summary())
print('Metrics for train: ', model.evaluate(X_train, Y_train))
print('Metrics for test: ', model.evaluate(X_test, Y_test))

# Linear regression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print('MSE = ', mse)

