# importing libraries
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# --------------------------------------------------------------------------------------

# loading data
x = load_breast_cancer()
df = pd.DataFrame(x.data, columns=[x.feature_names])
names = x["feature_names"]
print(names)
print(df.head())

df.info()
# defining X and Y
X = df.values
Y = x['target']

# creating test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
print("X_train set: ", X_train)
print("y_test set: ", y_test)
# # --------------------------------------------------------------------------------------
#
# -- LOGISTIC REGRESSION -- #
# classifier = LogisticRegression(max_iter=4000, solver="lbfgs")
# classifier = classifier.fit(X_train, y_train)
#
# # scores for train set
# predicted = classifier.predict(X_train)
# print('Accuracy:', accuracy_score(y_train, predicted))
# print('F1:', f1_score(y_train, predicted))
#
# # scores for test set
# predicted = classifier.predict(X_test)
# print('Accuracy:', accuracy_score(y_test, predicted))
# print('F1:', f1_score(y_test, predicted))
#
# # report for test set
# print(classification_report(y_test, predicted))
# # ----------------------------------------------------------------------------
#
# # -- K NEIGHBOURS -- #
# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier.fit(X_train, y_train)
#
# # scores for train set
# predicted = classifier.predict(X_train)
# print('Accuracy:', accuracy_score(y_train, predicted))
# print('F1:', f1_score(y_train, predicted))
#
# # scores for test set
# predicted = classifier.predict(X_test)
# print('Accuracy:', accuracy_score(y_test, predicted))
# print('F1:', f1_score(y_test, predicted))
#
# # report for test set
# print(classification_report(y_test, predicted))
# # --------------------------------------------------------------------------------
#
# # -- DECISION TREE -- #
# dt_gini = DecisionTreeClassifier(criterion="gini")
# dt_gini.fit(X_train, y_train)
# dt_entropy = DecisionTreeClassifier(criterion="entropy")
# dt_entropy.fit(X_train, y_train)
#
# # scores for train set (gini)
# gini_predicted = dt_gini.predict(X_train)
# print('Accuracy:', accuracy_score(y_train, gini_predicted))
# print('F1:', f1_score(y_train, gini_predicted))
#
# # scores for train set (entropy)
# entropy_predicted = dt_entropy.predict(X_train)
# print('Accuracy:', accuracy_score(y_train, entropy_predicted))
# print('F1:', f1_score(y_train, entropy_predicted))
#
# # scores for test set (gini)
# gini_predicted = dt_gini.predict(X_test)
# print('Accuracy:', accuracy_score(y_test, gini_predicted))
# print('F1:', f1_score(y_test, gini_predicted))
#
# # scores for test set (entropy)
# entropy_predicted = dt_entropy.predict(X_test)
# print('Accuracy:', accuracy_score(y_test, entropy_predicted))
# print('F1:', f1_score(y_test, entropy_predicted))
# # ----------------------------------------------------------------------------
#
# # -- RANDOM FOREST -- #
# classifier = RandomForestClassifier(n_estimators=100)
# classifier.fit(X_train, y_train)
#
# # scores for train set
# predicted = classifier.predict(X_train)
# print('Accuracy:', accuracy_score(y_train, predicted))
# print('F1:', f1_score(y_train, predicted))
#
# # scores for test set
# predicted = classifier.predict(X_test)
# print('Accuracy:', accuracy_score(y_test, predicted))
# print('F1:', f1_score(y_test, predicted))
#
# # report for test set
# print(classification_report(y_test, predicted))
# # -----------------------------------------------------------------------------
#
# # -- ADA BOOST -- #
# classifier = AdaBoostClassifier()
# classifier.fit(X_train, y_train)
#
# # scores for train set
# predicted = classifier.predict(X_train)
# print('Accuracy:', accuracy_score(y_train, predicted))
# print('F1:', f1_score(y_train, predicted))
#
# # scores for test set
# predicted = classifier.predict(X_test)
# print('Accuracy:', accuracy_score(y_test, predicted))
# print('F1:', f1_score(y_test, predicted))
#
# # report for test set
# print(classification_report(y_test, predicted))
# # -----------------------------------------------------------------------------
#
# # SVC
# classifier = SVC()
# classifier.fit(X_train, y_train)
#
# # scores for train set
# predicted = classifier.predict(X_train)
# print('Accuracy:', accuracy_score(y_train, predicted))
# print('F1:', f1_score(y_train, predicted))
#
# # scores for test set
# predicted = classifier.predict(X_test)
# print('Accuracy:', accuracy_score(y_test, predicted))
# print('F1:', f1_score(y_test, predicted))
#
# # report for test set
# print(classification_report(y_test, predicted))
# # -----------------------------------------------------------------------------
#
# # -- MLP -- #
# classifier = MLPClassifier()
# classifier.fit(X_train, y_train)
#
# # scores for train set
# predicted = classifier.predict(X_train)
# print('Accuracy:', accuracy_score(y_train, predicted))
# print('F1:', f1_score(y_train, predicted))
#
# # scores for test set
# predicted = classifier.predict(X_test)
# print('Accuracy:', accuracy_score(y_test, predicted))
# print('F1:', f1_score(y_test, predicted))
#
# # report for test set
# print(classification_report(y_test, predicted))
# # ----------------------------------------------------------------------------
#
# classifiers = [LogisticRegression(max_iter=1000), KNeighborsClassifier(n_neighbors=5),
#                DecisionTreeClassifier(criterion="gini"), DecisionTreeClassifier(criterion="entropy"),
#                RandomForestClassifier(n_estimators=100), AdaBoostClassifier(), SVC(), MLPClassifier()]
#
# for x in range(len(classifiers)):
#     print(classifiers[x])
