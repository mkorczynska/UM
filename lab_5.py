from keras import Sequential
from keras.layers import Dense
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn

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

# ----------------------------------------------
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import random as rm
from keras.losses import sparse_categorical_crossentropy

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.show()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# num_classes = y_test.shape[1]

model = Sequential()
model.add(Conv2D(28, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss=sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

los = rm.randint(0, 10000)
predictions = model.predict(X_test)
print("Random number: ", los)
print("Prediction: ", predictions[los])
print("Y: ", y_test[los])

(X_train, y_train), (X_test, y_test) = mnist.load_data()
plt.imshow(X_test[los], cmap=plt.get_cmap('gray'))
plt.show()
