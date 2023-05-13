import os
import numpy as np 
import pandas as pd 
import warnings
import matplotlib.pyplot as plt
from sklearn import datasets
from pandas.plotting import scatter_matrix
from sklearn import linear_model
from sklearn import neighbors
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
    np.random.seed(100)
    X, y = datasets.make_friedman1(1000)
    
    ols = linear_model.LinearRegression()
    elastic = linear_model.ElasticNetCV()
    ridge = linear_model.RidgeCV()
    lars = linear_model.LarsCV()
    lasso = linear_model.LassoCV()
    knn = neighbors.KNeighborsRegressor()
    
    superLeaner = sl.SuperLearner([ols, elastic, ridge, lars, lasso, knn])
    
    superLeaner.fit(X, y)
    y_pred = superLeaner.predict(X)
    
    print(superLeaner.error)
    print(superLeaner.weights)
    
    
if __name__ == "__main__":
    main()