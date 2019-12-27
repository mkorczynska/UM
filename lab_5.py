# from keras import Sequential
# from keras.layers import Dense
# import pandas
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# import sklearn
#
# # read dataset
# dataset = pandas.read_csv('C:/Users/HP ENVY/Downloads/Weather.csv', low_memory=False)
# print(dataset.shape)
# print(dataset.describe())
#
# X = dataset['MinTemp'].values.reshape(-1, 1)
# Y = dataset['MaxTemp'].values.reshape(-1, 1)
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#
# # MLP
# model = Sequential()
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
# model.fit(X_train, Y_train, epochs=10)
# print(model.summary())
# print('Metrics for train: ', model.evaluate(X_train, Y_train))
# print('Metrics for test: ', model.evaluate(X_test, Y_test))
#
# # Linear regression
# regressor = LinearRegression()
# regressor.fit(X_train, Y_train)
# Y_pred = regressor.predict(X_test)
# mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
# print('MSE = ', mse)

# ----------------------------------------------
from keras.datasets import mnist

# download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

# plot the first image in the dataset
plt.imshow(X_train[0])

# check image shape
X_train[0].shape

# reshape data to fit model
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

from keras.utils import to_categorical

# one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train[0]

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# create model
model = Sequential()
# add model layers
model.add(Conv2D(28, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(10, activation='softmax'))

# compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

#predict first 4 images in the test set
model.predict(X_test[:4])

#actual results for first 4 images in test set
y_test[:4]
