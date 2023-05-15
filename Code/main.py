#Autor: Dragos Tanasa
from sklearn import datasets
from sklearn import linear_model
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    beta = np.random.randint(low = -20, high = 20, size = 10)
    #random set some beta to 0
    beta = np.where(np.random.randint(low = 0, high = 2, size = 10) == 0, 0, beta)
    print(beta)
    
    # Library of base estimators
    library = {
        "osl": linear_model.LinearRegression(),
        "ridge": linear_model.RidgeCV(alphas=np.arange(0.1, 5, 0.1)),
        "lasso": linear_model.LassoCV(alphas=np.arange(0.1, 5, 0.1)),
        "enet": linear_model.ElasticNetCV(alphas=np.arange(0.1, 5, 0.1), l1_ratio=np.arange(0.1, 1, 0.1)),
        "knn_5": neighbors.KNeighborsRegressor(n_neighbors=5),
        "knn_10": neighbors.KNeighborsRegressor(n_neighbors=10),
        "knn_15": neighbors.KNeighborsRegressor(n_neighbors=15),
    }
    
    # Initialize the error matrix
    error = np.empty((sim, len(library)+1))
    
    sample_sizes = [100, 200, 500, 1000, 5000, 10000]
    
    for n in sample_sizes:
    
        for i in tqdm.tqdm(range(sim), desc="Simulation with {n} samples... ".format(n=n)):
            X, y = ms.make_regression_fixed_coeffs(n_samples = n, n_features = len(beta), coefficients = beta, noise = 20)
            #X, y = datasets.make_regression(n_samples = n, n_features = len(beta), noise = 30, random_state=2)
            #X, y = datasets.make_friedman1(n)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
            y_test = scaler.fit_transform(y_test.reshape(-1,1)).flatten()

            for estimator in library:
                library[estimator].fit(X_train, y_train)

            sl1 = psl.SuperLearner(library)
            sl1.fit(X_train, y_train)

            for j, estimator in enumerate(library):
                error[i, j] = library[estimator].score(X_test, y_test)
            error[i, len(library)] = sl1.score(X_test, y_test)

        df = pd.DataFrame(error, columns=list(library.keys()) + ["SuperLearner"])
        df.to_csv(path + "/Data/R2_at_{n}.csv".format(n=n), index=False)       
        
    print("Done!")
    
    print("super learner weights:", sl1.weights)
        
if __name__ == "__main__":
    montecarlo()
    