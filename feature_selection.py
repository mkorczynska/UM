# importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression  # F-value between label/feature for regression tasks.
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
# from boruta import BorutaPy
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

x = load_boston()
df = pd.DataFrame(x.data, columns=x.feature_names)
df["MEDV"] = x.target
X = df.drop("MEDV", 1)  # Feature Matrix
y = df["MEDV"]  # Target Variable
df.head()

###### Pearson Correlation ########

plt.figure(figsize=(12, 10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
# Correlation with output variable
cor_target = abs(cor["MEDV"])
# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.5]
print(relevant_features)
print(df[["LSTAT", "PTRATIO"]].corr())
print(df[["RM", "LSTAT"]].corr())

###### Variance Threshold ########
threshold_n = 0.95
selector = VarianceThreshold(threshold=(threshold_n * (1 - threshold_n)))
selected = selector.fit_transform(X)
# pd.set_option('display.max_columns', None)
# print(X[X.columns[selector.get_support(indices=True)]])
print(X.columns[selector.get_support()].values)

###### KBest ########

selector = SelectKBest(f_regression, 4)
selected = selector.fit_transform(X, y)
print(X.columns[selector.get_support()].values)

###### p-value ########

# Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(X)
# Fitting sm.OLS model
model = sm.OLS(y, X_1).fit()
print(model.pvalues)

###### Backward Elimination ########

cols = list(X.columns)
pmax = 1
while (len(cols) > 0):
    p = []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y, X_1).fit()
    p = pd.Series(model.pvalues.values[1:], index=cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if (pmax > 0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

###### RFE ########

model = LinearRegression()
# Initializing RFE model
rfe = RFE(model, 7)
# Transforming data using RFE
X_rfe = rfe.fit_transform(X, y)
# Fitting the data to model
model.fit(X_rfe, y)
print(rfe.support_)
print(rfe.ranking_)

# no of features
nof_list = np.arange(1, 13)
high_score = 0
# Variable to store the optimum features
nof = 0
score_list = []
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LinearRegression()
    rfe = RFE(model, nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe, y_train)
    score = model.score(X_test_rfe, y_test)
    score_list.append(score)
    if (score > high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" % nof)
print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)
model = LinearRegression()
# Initializing RFE model
rfe = RFE(model, 10)
# Transforming data using RFE
X_rfe = rfe.fit_transform(X, y)
# Fitting the data to model
model.fit(X_rfe, y)
temp = pd.Series(rfe.support_, index=cols)
selected_features_rfe = temp[temp == True].index
print(selected_features_rfe.values)

###### Embedded ########

lasso = Lasso()
featureSelection = SelectFromModel(lasso)
featureSelection.fit(X, y)
selectedFeatures = featureSelection.transform(X)
print(X.columns[featureSelection.get_support()].values)

clf = Lasso()

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]
while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]
print(X.columns[sfm.get_support()].values)

rf = RandomForestRegressor()
rf.fit(X, y)
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index=X_train.columns, columns=['importance']).sort_values('importance',
                                                                                              ascending=False)
print(feature_importances)

# Random Forests for Boruta
rf_boruta = RandomForestRegressor(n_jobs=-1)
# Perform Boruta
boruta = BorutaPy(rf_boruta, n_estimators='auto', verbose=2)
boruta.fit(X_train.values, y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
print(X.columns[rfe.support_].values)
