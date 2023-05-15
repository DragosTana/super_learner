#Autor: Dragos Tanasa
from sklearn import datasets
from sklearn import linear_model
from sklearn import neighbors
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import tqdm

import misc as ms
import parallelSuperLearner as sl

def main():
    pass
    
def montecarlo():
    
    sim = 1000000
    n = 100
    beta = np.ones(5)
    
    library2 = {
        "ols": linear_model.LinearRegression(),
        "ridge": linear_model.RidgeCV(alphas=np.arange(0.01, 10.0, 0.01)),
        "lasse" : linear_model.LassoCV(alphas=np.arange(0.01, 10.0, 0.01), positive=True),
        "elastic" :  linear_model.ElasticNetCV(alphas=np.arange(0.01, 10.0, 0.01), positive=True),
        "knn_5": neighbors.KNeighborsRegressor(n_neighbors=5),
        "knn_10": neighbors.KNeighborsRegressor(n_neighbors=10),
        "knn_15": neighbors.KNeighborsRegressor(n_neighbors=15),
        "knn_20": neighbors.KNeighborsRegressor(n_neighbors=20),
    }
        
    for i in tqdm.tqdm(range(sim), desc="Running montecarlo simulations... "):
        np.random.seed(2)
        X, y = ms.make_regression_fixed_coeffs(n_samples = n, n_features = len(beta), coefficients = beta, noise = 1.0)
 
        
    
if __name__ == "__main__":
    montecarlo()