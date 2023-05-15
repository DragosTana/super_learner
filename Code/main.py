#Autor: Dragos Tanasa
from sklearn import datasets
from sklearn import linear_model
from sklearn import neighbors
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import os
import numpy as np 
import pandas as pd 
import tqdm

import misc as ms
import parallelSuperLearner as psl

def main():
    pass
    
def montecarlo():
    
    path = os.getcwd()
    
    # Parameters for the simulation: sim = number of simulations, n = sample size, beta = true coefficients
    sim = 100
    n = 500
    beta = np.ones(5)
    
    # Library of base estimators
    library = {
        "ols": linear_model.LinearRegression(),
        "ridge": linear_model.RidgeCV(alphas=np.arange(0.01, 10.0, 0.01)),
        "lasse" : linear_model.LassoCV(alphas=np.arange(0.01, 10.0, 0.01), positive=True),
        "elastic" :  linear_model.ElasticNetCV(alphas=np.arange(0.01, 10.0, 0.01), positive=True),
        "knn_5": neighbors.KNeighborsRegressor(n_neighbors=5),
        "knn_10": neighbors.KNeighborsRegressor(n_neighbors=10),
        "knn_15": neighbors.KNeighborsRegressor(n_neighbors=15),
        "knn_20": neighbors.KNeighborsRegressor(n_neighbors=20),
    }
    
    # Initialize the error matrix
    error = np.empty((sim, len(library)+1))
    
    for i in tqdm.tqdm(range(sim), desc="Simulation with {n} samples... ".format(n=n)):
        np.random.seed(2)
        X, y = ms.make_regression_fixed_coeffs(n_samples = n, n_features = len(beta), coefficients = beta, noise = 1.0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
        
        for estimator in library:
            library[estimator].fit(X_train, y_train)
            
        sl1 = psl.SuperLearner(library)
        sl1.fit(X_train, y_train)
        
        for j, estimator in enumerate(library):
            error[i, j] = library[estimator].score(X_test, y_test)
        error[i, len(library)] = sl1.score(X_test, y_test)
    
    np.savetxt(path + "/Data/MSE_at_{n}.csv".format(n=n), error, delimiter=",")        
        
        
if __name__ == "__main__":
    montecarlo()