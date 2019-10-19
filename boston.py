import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
# from sklearn.feature_selectionimport VarianceThreshold
from sklearn import metrics

from sklearn.datasets import load_boston

boston = load_boston()
print(boston.data.shape)
print(boston)

x = load_boston()
df = pd.DataFrame(x.data, columns=x.feature_names)
df["MEDV"] = x.target
X = df.drop("MEDV", 1)
y = df["MEDV"]
df.head()
print(df.head())

plt.figure(figsize=(12, 10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()


# Correlation with output variable
cor_target = abs(cor["MEDV"])
# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.5]

print(df[["LSTAT", "PTRATIO"]].corr())
print(df[["RM", "LSTAT"]].corr())

# print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
boston_lr = LinearRegression()
boston_lr.fit(X_train, y_train)
print(boston_lr.coef_)
print(boston_lr.intercept_)
print('R2 for train: ', boston_lr.score(X_train, y_train))
print('R2 for test: ', boston_lr.score(X_test, y_test))

y_pred = boston_lr.predict(X_test)
df = pd.DataFrame({'actual': y_test, 'pred': y_pred})
print(df)

print(pd.DataFrame(boston_lr.coef_))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# -------------------------------------
boston_rr = Ridge()
boston_rr.fit(X_train, y_train)
print(boston_rr.coef_)
print(boston_rr.intercept_)
print('R2 for train: ', boston_rr.score(X_train, y_train))
print('R2 for test: ', boston_rr.score(X_test, y_test))

y_pred = boston_rr.predict(X_test)
df = pd.DataFrame({'actual': y_test, 'pred': y_pred})
print(df)

print(pd.DataFrame(boston_rr.coef_))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# -------------------------------------
boston_l = Lasso()
boston_l.fit(X_train, y_train)
print(boston_l.coef_)
print(boston_l.intercept_)
print('R2 for train: ', boston_l.score(X_train, y_train))
print('R2 for test: ', boston_l.score(X_test, y_test))

y_pred = boston_l.predict(X_test)
df = pd.DataFrame({'actual': y_test, 'pred': y_pred})
print(df)

print(pd.DataFrame(boston_l.coef_))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

pd.Series(boston_l.coef_)
plt.show()
