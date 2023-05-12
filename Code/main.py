import os
import numpy as np 
import pandas as pd 
import warnings
import sklearn
import matplotlib.pyplot as plt
from sklearn import datasets
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression, ElasticNetCV, Ridge
from sklearn.model_selection import train_test_split

import superLearner as sl

def generate_data(features = 10, n = 1000, plot = True):
    
    #parameters 
    beta = np.random.uniform(-20, 20, size = features).astype(int)
    size = n
    
    #generate covariance matrix 
    mean = np.random.uniform(-5, 5, size = features).astype(int)
    cov = datasets.make_spd_matrix(features, random_state=1)
                
    #data generating process
    X = np.random.multivariate_normal(mean, cov, size=size)
    
    if plot:
        #transforming data to pandas dataframe and plotting correlation matrix and scatter matrix
        df = pd.DataFrame(X, columns = ["X{i}".format(i=i) for i in range(1, features+1)])
        f = plt.figure()
        plt.matshow(df.corr(), fignum=f.number, cmap='coolwarm')
        plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
        plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16)

        scatter_matrix(df, alpha = 0.2,  figsize = (6, 6), diagonal = 'kde')

        plt.show()
    
    Y = np.dot(X, beta) + np.random.normal(0, 10, size = n)

    return Y, X

def main():
    Y, X = generate_data(features = 5, n = 100, plot=False)
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

    lasso = LassoCV(cv=5)
    ridge = RidgeCV(cv=5)
    l = LinearRegression()
    el = ElasticNetCV(cv=5)
    
    models = [lasso, ridge, l, el]
    
    sl_model = sl.SuperLearner(models)
    
    sl_model.fit(x_train, y_train)
    y = sl_model.predict(x_test)
    
    print("R2 sl_model: ", sklearn.metrics.r2_score(y_test, y))
    print("R2 lasso: ", sklearn.metrics.r2_score(y_test, lasso.fit(x_train, y_train).predict(x_test)))
    print("R2 ridge: ", sklearn.metrics.r2_score(y_test, ridge.fit(x_train, y_train).predict(x_test)))
    print("R2 l: ", sklearn.metrics.r2_score(y_test, l.fit(x_train, y_train).predict(x_test)))
    print("R2 el: ", sklearn.metrics.r2_score(y_test, el.fit(x_train, y_train).predict(x_test)))
    print("coef: ", sl_model.weights)
    
if __name__ == "__main__":
    main()