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
import time

import misc as ms
import parallelSuperLearner as psl
import superLearner as sl

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
        "ols": linear_model.LinearRegression(),
        "elastic_0.01": linear_model.ElasticNet(alpha=0.01),
        "elastic_0.1": linear_model.ElasticNet(alpha=0.1),
        "elastic_1.0": linear_model.ElasticNet(alpha=1.0),
        "elastic_10.0": linear_model.ElasticNet(alpha=10.0),
        "ridge_0.01": linear_model.Ridge(alpha=0.01),
        "ridge_0.1": linear_model.Ridge(alpha=0.1),
        "ridge_1.0": linear_model.Ridge(alpha=1.0),
        "ridge_10.0": linear_model.Ridge(alpha=10.0),
        "lasso_0.01": linear_model.Lasso(alpha=0.01),
        "lasso_0.1": linear_model.Lasso(alpha=0.1),
        "lasso_1.0": linear_model.Lasso(alpha=1.0),
        "lasso_10.0": linear_model.Lasso(alpha=10.0),
        "knn_5": neighbors.KNeighborsRegressor(n_neighbors=5),
        "knn_10": neighbors.KNeighborsRegressor(n_neighbors=10),
        "knn_15": neighbors.KNeighborsRegressor(n_neighbors=15),
        "knn_20": neighbors.KNeighborsRegressor(n_neighbors=20),
    }
    
    # Initialize the error matrix
    error = np.empty((sim, len(library)+1))
    
    sample_sizes = [100, 200, 500, 1000, 5000, 10000]
    
    for n in sample_sizes:
    
        for i in tqdm.tqdm(range(sim), desc="Simulation with {n} samples... ".format(n=n)):
            X, y = ms.make_regression_fixed_coeffs(n_samples = n, n_features = len(beta), coefficients = beta, noise = 10)
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
    
def speedUp():
    
    sample = [100, 200, 500, 1000, 2000, 5000, 7500, 10000]
    lib_size = [5, 10, 20, 50, 80, 100]
    
    times_sl = np.zeros((len(sample), len(lib_size)))
    times_psl = np.zeros((len(sample), len(lib_size)))
    beta = np.random.randint(low = -20, high = 20, size = 10)

    library = {}
    

    for j in tqdm.tqdm(range(len(lib_size))):
        
        library.clear()
        #creaiamo la libreria
        for q in range(lib_size[j]):
            library["ols_{}".format(q)] = linear_model.LinearRegression()
            
        for i in range(len(sample)):
            
            sl1 = sl.SuperLearner(library)
            psl1 = psl.SuperLearner(library)
            mean_time_sl = 0
            mean_time_psl = 0
            
            for t in range(10):
                #creiamo il dataset
                X, y = ms.make_regression_fixed_coeffs(n_samples = sample[i], n_features = len(beta), coefficients = beta, noise = 10)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.fit_transform(X_test)
                y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
                y_test = scaler.fit_transform(y_test.reshape(-1,1)).flatten()
                
                start_time = time.time()
                sl1.fit(X_train, y_train)
                endtime = time.time()
                mean_time_sl += endtime - start_time
                
                start_time = time.time()
                psl1.fit(X_train, y_train)
                endtime = time.time()
                mean_time_psl += endtime - start_time
                
            times_sl[i, j] = mean_time_sl / 10
            times_psl[i, j] = mean_time_psl / 10
    
    np.savetxt("times_sl.csv", times_sl, delimiter=",")
    np.savetxt("times_psl.csv", times_psl, delimiter=",")
    
    speed_up = times_sl / times_psl
    print(speed_up)
                    
if __name__ == "__main__":
    speedUp()
