import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set_style("darkgrid")
boston = datasets.load_boston()
boston_df = pd.DataFrame(data=boston['data'], columns=boston['feature_names'])
print(boston_df)
n_houses = boston_df.shape[0]
rand_noise = np.random.rand(n_houses, 5)

rand_noise_df = pd.DataFrame(data=rand_noise, columns=['Noise_1', 'Noise_2', 'Noise_3', 'Noise_4', 'Noise_5'])
X = pd.concat([boston_df, rand_noise_df], axis=1)
y = boston['target']

boston_lr = LinearRegression()
boston_ls = Lasso()
boston_rg = Ridge()
boston_en = ElasticNet()
models = [(boston_lr, 'Linear Regression'),
           (boston_ls, 'Lasso'),
           (boston_rg, "Ridge Regression"),
           (boston_en, "Elastic Net")]

boston_ss = StandardScaler()
X_scaled = boston_ss.fit_transform(X=X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=235)

for model in models:
    model[0].fit(X_train, y_train)
    pd.Series(model[0].coef_,
              index=X.columns).plot(kind='barh')
    plt.title(model[1])
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature Name')
    plt.show()

for model in models:
    mse = mean_squared_error(model[0].predict(X_test), y_test)
    print(f"Mean Squared Error of {model[1]}: {mse:.2f}")