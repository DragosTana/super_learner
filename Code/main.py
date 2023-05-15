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
    beta = np.random.randint(low = -100, high = 100, size = 10)
    
    # Library of base estimators
    library = {
        "osl": linear_model.LinearRegression(),
        "ridge": linear_model.Ridge(alpha=0.5),
        "lasso": linear_model.Lasso(alpha=0.5),
        "enet": linear_model.ElasticNet(alpha=0.5),
        "knn": neighbors.KNeighborsRegressor(n_neighbors=5),
    }
    
    # Initialize the error matrix
    error = np.empty((sim, len(library)+1))
    
    sample_sizes = [100, 200, 500, 1000, 5000, 10000]
    
    for n in sample_sizes:
    
        for i in tqdm.tqdm(range(sim), desc="Simulation with {n} samples... ".format(n=n)):
            np.random.seed(2)
            X, y = ms.make_regression_fixed_coeffs(n_samples = n, n_features = len(beta), coefficients = beta, noise = 10)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
            
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
    