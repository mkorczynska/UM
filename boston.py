import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
# ---------------------------------------------------------

# loading dataset boston
from sklearn.datasets import load_boston

x = load_boston()
print("Boston data shape: ", x.data.shape)

names = x["feature_names"]
# print(names)
df = pd.DataFrame(x.data, columns=x.feature_names)
df["MEDV"] = x.target

print("Boston data frame: ", df)

# defining x and y
X = df.drop("MEDV", 1)
Y = df["MEDV"]
# df.head()
print(df.head())

# # plot
# plt.figure(figsize=(12, 10))
# cor = df.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# # plt.show()
#
# # Correlation with output variable
# cor_target = abs(cor["MEDV"])
# # Selecting highly correlated features
# relevant_features = cor_target[cor_target > 0.5]
#
# print(df[["LSTAT", "PTRATIO"]].corr())
# print(df[["RM", "LSTAT"]].corr())
#
# # print(X)

# creating test set and train set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
print("X_train set: ", X_train)
print("X_test set: ", X_test)

# --- LINEAR REGRESSION --- #
boston_lr = LinearRegression()
boston_lr.fit(X_train, y_train)
print("Coefficients: ", boston_lr.coef_)
print("Intercept: ", boston_lr.intercept_)

# R for train and test set
print('R2 for train: ', boston_lr.score(X_train, y_train))
print('R2 for test: ', boston_lr.score(X_test, y_test))
#
# linear regression - prediction
y_pred = boston_lr.predict(X_test)
df = pd.DataFrame({'actual': y_test, 'pred': y_pred})
print("Prediction: ")
print(df)

print(pd.DataFrame(boston_lr.coef_))

# # errors
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# # -------------------------------------

# --- RIDGE REGRESSION ---  #
boston_rr = Ridge()
boston_rr.fit(X_train, y_train)
print("Coefficients: ", boston_rr.coef_)
print("Intercept: ", boston_rr.intercept_)

# R for train and test set
print('R2 for train: ', boston_rr.score(X_train, y_train))
print('R2 for test: ', boston_rr.score(X_test, y_test))

# ridge regression - prediction
y_pred = boston_rr.predict(X_test)
df = pd.DataFrame({'actual': y_test, 'pred': y_pred})
print(df)

print(pd.DataFrame(boston_rr.coef_))

# errors
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# -------------------------------------

print("LASSO")
# --- LASSO --- #
boston_l = Lasso()
boston_l.fit(X_train, y_train)
print("Coefficients: ", boston_l.coef_)
print("Intercept: ", boston_l.intercept_)

# R for train and test set
print('R2 for train: ', boston_l.score(X_train, y_train))
print('R2 for test: ', boston_l.score(X_test, y_test))

# lasso - prediction
y_pred = boston_l.predict(X_test)
df = pd.DataFrame({'actual': y_test, 'pred': y_pred})
print(df)

print(pd.DataFrame(boston_l.coef_))

# errors
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# --- ELASTIC NET --- #
boston_en = ElasticNet()
boston_en.fit(X_train, y_train)
print("Coefficients: ", boston_en.coef_)
print("Intercept: ", boston_en.intercept_)

# R for train and test set
print('R2 for train: ', boston_en.score(X_train, y_train))
print('R2 for test: ', boston_en.score(X_test, y_test))

# elastic net- prediction
y_pred = boston_l.predict(X_test)
df = pd.DataFrame({'actual': y_test, 'pred': y_pred})
print(df)

print(pd.DataFrame(boston_l.coef_))

# errors
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# pd.Series(boston_l.coef_)
# plt.show()
# # -----------------------------------

# --- VARIANCE THRESHOLD --- #
print("THRESHOLD")
t = 0.95  # threshold
selector = VarianceThreshold(threshold=t * (1 - t))
selected = selector.fit_transform(X)

# printing columns
print(X.columns[selector.get_support()].values)

X_train = X_train.drop("NOX", 1)
X_test = X_test.drop("NOX", 1)

boston_lr_t = LinearRegression()
boston_lr_t.fit(X_train, y_train)
print(boston_lr_t.coef_)
print(boston_lr_t.intercept_)
print('R2 for train: ', boston_lr_t.score(X_train, y_train))
print('R2 for test: ', boston_lr_t.score(X_test, y_test))

y_pred = boston_lr_t.predict(X_test)
df = pd.DataFrame({'actual': y_test, 'pred': y_pred})
print(df)

print(pd.DataFrame(boston_lr_t.coef_))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# --- RANDOM FORREST --- #
boston_rf = RandomForestRegressor()
boston_rf.fit(X, Y)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), boston_rf.feature_importances_), names), reverse=True))