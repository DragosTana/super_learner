#Autor: Dragos Tanasa
from sklearn import datasets
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import os
import numpy as np 
import pandas as pd 
import tqdm
import time
import random

import misc as ms
import parallelSuperLearner as psl
import superLearner as sl

def main():
    pass
    
def montecarloSlvsLib():
    
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
        "ridge": linear_model.RidgeCV(alphas=np.arange(0.1, 10, 0.1)),
        "lasso" : linear_model.LassoCV(alphas=np.arange(0.1, 10.0, 0.1)),
        "elastic" :  linear_model.ElasticNetCV(alphas=np.arange(0.1, 10.0, 0.1)),
        "knn_5": neighbors.KNeighborsRegressor(n_neighbors=5),
        "knn_10": neighbors.KNeighborsRegressor(n_neighbors=10),
        "knn_15": neighbors.KNeighborsRegressor(n_neighbors=15),
        "random_forest": ensemble.RandomForestRegressor(n_estimators=50),
        "gradient_boosting": ensemble.GradientBoostingRegressor(n_estimators=50),
    }
    
    # Initialize the error matrix
    error = np.empty((sim, len(library)+1))
    
    sample_sizes = [100, 200, 500, 1000, 5000, 10000, 15000, 20000]
    
    for n in sample_sizes:
    
        for i in tqdm.tqdm(range(sim), desc="Simulation with {n} samples... ".format(n=n)):
            #X, y = ms.make_regression_fixed_coeffs(n_samples = n, n_features = len(beta), coefficients = beta, noise = 10)
            #X, y = datasets.make_regression(n_samples = n, n_features = len(beta), noise = 30, random_state=2)
            X, y = datasets.make_friedman1(n)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
            y_test = scaler.fit_transform(y_test.reshape(-1,1)).flatten()

            for estimator in library:
                library[estimator].fit(X_train, y_train)

            sl1 = psl.SuperLearner(library, meta_learner=linear_model.ElasticNetCV(alphas=np.arange(0.01, 10.0, 0.01), l1_ratio=np.arange(0.1, 1, 0.1), positive=True))
            sl1.fit(X_train, y_train)

            for j, estimator in enumerate(library):
                error[i, j] = library[estimator].score(X_test, y_test)
            error[i, len(library)] = sl1.score(X_test, y_test) 

        df = pd.DataFrame(error, columns=list(library.keys()) + ["SuperLearner"])
        df.to_csv(path + "/Data/R2_at_{n}.csv".format(n=n), index=False)       
        
    print("Done!")
    
    print("super learner weights:", sl1.weights)
    
def montecarloLibrary():
    path = os.getcwd()
    # Parameters for the simulation: sim = number of simulations, n = sample size, beta = true coefficients
    sim = 100
    beta = np.random.randint(low = -20, high = 20, size = 10)
    #random set some beta to 0
    beta = np.where(np.random.randint(low = 0, high = 2, size = 10) == 0, 0, beta)
    print(beta)
    sample_sizes = [100, 1000, 5000]
    
    library1 = {
        "ols": linear_model.LinearRegression(),
        "ridge": linear_model.RidgeCV(alphas=np.arange(0.1, 10, 0.1)),
        "lasso" : linear_model.LassoCV(alphas=np.arange(0.1, 10.0, 0.1)),
        "elastic" :  linear_model.ElasticNetCV(alphas=np.arange(0.1, 10.0, 0.1)),
    }
    
    library2 = {
        "ols": linear_model.LinearRegression(),
        "ridge_0.1": linear_model.Ridge(alpha=0.1),
        "ridge_0.5": linear_model.Ridge(alpha=0.5),
        "ridge_1": linear_model.Ridge(alpha=1),
        "ridge_2": linear_model.Ridge(alpha=2),
        "ridge_5": linear_model.Ridge(alpha=5),
        "ridge_10": linear_model.Ridge(alpha=10),
        "lasso_0.1": linear_model.Lasso(alpha=0.1),
        "lasso_0.5": linear_model.Lasso(alpha=0.5),
        "lasso_1": linear_model.Lasso(alpha=1),
        "lasso_2": linear_model.Lasso(alpha=2),
        "lasso_5": linear_model.Lasso(alpha=5),
        "lasso_10": linear_model.Lasso(alpha=10),
        "elastic_0.1": linear_model.ElasticNet(alpha=0.1),
        "elastic_0.5": linear_model.ElasticNet(alpha=0.5),
        "elastic_1": linear_model.ElasticNet(alpha=1),
        "elastic_2": linear_model.ElasticNet(alpha=2),
        "elastic_5": linear_model.ElasticNet(alpha=5),
        "elastic_10": linear_model.ElasticNet(alpha=10),
    }
    
    error = np.empty((sim, 2))
    for n in sample_sizes:

        for i in tqdm.tqdm(range(sim), desc="Simulation with {n} samples... ".format(n=n)):
            X, y = ms.make_regression_fixed_coeffs(n_samples = n, n_features = len(beta), coefficients = beta, noise = 10)
            #X, y = datasets.make_regression(n_samples = n, n_features = len(beta), noise = 30, random_state=2)
            #X, y = datasets.make_friedman1(n)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
            y_test = scaler.fit_transform(y_test.reshape(-1,1)).flatten()

            for estimator in library1:
                library1[estimator].fit(X_train, y_train)
                
            for estimator in library2:
                library2[estimator].fit(X_train, y_train)
                
            sl1 = psl.SuperLearner(library1)
            sl2 = psl.SuperLearner(library2)
            
            sl1.fit(X_train, y_train)
            sl2.fit(X_train, y_train)
            
            error[i, 0] = sl1.score(X_test, y_test)
            error[i, 1] = sl2.score(X_test, y_test)
        
        df = pd.DataFrame(error, columns=["FewOptimized", "LotsNotOptimized"])
        df.to_csv(path + "/Data/R2_at_{n}_lib.csv".format(n=n), index=False)
    
def montecarloFoldNumber():
    
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
        "ridge": linear_model.RidgeCV(alphas=np.arange(0.1, 10, 0.1)),
        "lasso" : linear_model.LassoCV(alphas=np.arange(0.1, 10.0, 0.1)),
        "elastic" :  linear_model.ElasticNetCV(alphas=np.arange(0.1, 10.0, 0.1)),
        "knn_5": neighbors.KNeighborsRegressor(n_neighbors=5),
        "knn_10": neighbors.KNeighborsRegressor(n_neighbors=10),
        "knn_15": neighbors.KNeighborsRegressor(n_neighbors=15),
    }
    
    # Initialize the error matrix
    error = np.empty((sim, 3))
    
    sample_sizes = [100, 200, 500, 1000, 5000, 10000, 15000]
    
    for n in sample_sizes:
    
        for i in tqdm.tqdm(range(sim), desc="Simulation with {n} samples... ".format(n=n)):
            #X, y = ms.make_regression_fixed_coeffs(n_samples = n, n_features = len(beta), coefficients = beta, noise = 10)
            X, y = datasets.make_regression(n_samples = n, n_features = len(beta), noise = 30, random_state=2)
            #X, y = datasets.make_friedman1(n)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
            y_test = scaler.fit_transform(y_test.reshape(-1,1)).flatten()

            for estimator in library:
                library[estimator].fit(X_train, y_train)

            sl_5 = psl.SuperLearner(library, folds = 5)
            sl_10 = psl.SuperLearner(library, folds = 10)

            if n  >= 10000:
                sl_adaptive = psl.SuperLearner(library, folds = 5)
            elif n >= 5000:
                sl_adaptive = psl.SuperLearner(library, folds = 10)
            elif n >= 1000:
                sl_adaptive = psl.SuperLearner(library, folds = 20)
            elif n >= 500:
                sl_adaptive = psl.SuperLearner(library, folds = 30)
            elif n >= 30:
                sl_adaptive = psl.SuperLearner(library, folds = len(X_train))
        
            sl_5.fit(X_train, y_train)
            sl_10.fit(X_train, y_train)
            sl_adaptive.fit(X_train, y_train)
            
            error[i, 0] = sl_5.score(X_test, y_test)
            error[i, 1] = sl_10.score(X_test, y_test)
            error[i, 2] = sl_adaptive.score(X_test, y_test)
        
        df = pd.DataFrame(error, columns=["5-fold", "10-fold", "Adaptive"])
        df.to_csv(path + "/Data/R2_at_{n}_folds.csv".format(n=n), index=False)
            
      
def montecarloOptimization():
    
    path = os.getcwd()
    library = {
        "ols": linear_model.LinearRegression(),
        "ridge": linear_model.RidgeCV(alphas=np.arange(0.05, 15, 0.05)),
        "lasso" : linear_model.LassoCV(alphas=np.arange(0.05, 15.0, 0.05)),
        "elastic" :  linear_model.ElasticNetCV(alphas=np.arange(0.05, 15.0, 0.05)),
        "knn_5": neighbors.KNeighborsRegressor(n_neighbors=5),
        "knn_10": neighbors.KNeighborsRegressor(n_neighbors=10),
        "knn_15": neighbors.KNeighborsRegressor(n_neighbors=15),
    }
    
    weigths_opt = []
    weigths_meta = []
    scores = []
    sl_scores = []
    sim = 500
    
    for i in tqdm.tqdm(range(sim)):
        X, y = datasets.make_friedman1(1000, noise=7)
        #X, y = datasets.make_regression(n_samples = 1000, n_features = 20, n_informative=10,  noise = 30)    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
        y_test = scaler.fit_transform(y_test.reshape(-1,1)).flatten()
        
        sl1 = sl.SuperLearner(library, meta_learner=linear_model.RidgeCV(alphas=np.arange(0.05, 15, 0.05)))
        sl2 = sl.SuperLearner(library)
        
        sl1.fit(X_train, y_train)
        sl2.fit(X_train, y_train)
        
        iteration_score = []
        for estimator in library:
            library[estimator].fit(X_train, y_train)
            s = library[estimator].score(X_test, y_test)
            iteration_score.append(s)
        
        weigths_meta.append(sl1.weights)
        weigths_opt.append(sl2.weights)
        scores.append(iteration_score)    
        sl_scores.append([sl1.score(X_test, y_test), sl2.score(X_test, y_test)])
        
    weigths_meta = np.array(weigths_meta)
    weigths_opt = np.array(weigths_opt)
    scores = np.array(scores)
    sl_scores = np.array(sl_scores)
    
    pd_meta = pd.DataFrame(weigths_meta, columns=list(library.keys()))
    pd_opt = pd.DataFrame(weigths_opt, columns=list(library.keys()))
    pd_score = pd.DataFrame(scores, columns=list(library.keys()))
    pd_sl_score = pd.DataFrame(sl_scores, columns=["meta", "opt"])
    
    pd_meta.to_csv(path + "/Data/weights_meta_frid.csv", index=False)
    pd_opt.to_csv(path + "/Data/weights_opt_frid.csv", index=False)
    pd_score.to_csv(path + "/Data/scores.csv_frid", index=False)
    pd_sl_score.to_csv(path + "/Data/sl_scores_frid.csv", index=False)
    
def timeTest(type, sample, lib_size):
    
    times = np.zeros((len(sample), len(lib_size)))
    
    lib = {}
    for i in range(len(sample)):
        
        for j in range(len(lib_size)):
            
            # Create the library
            library1 = {
                "ols": linear_model.LinearRegression(),
                "elastic_0.01": linear_model.ElasticNet(alpha=0.01),
                "elastic_0.1": linear_model.ElasticNet(alpha=0.1),
                "elastic_1.0": linear_model.ElasticNet(alpha=1.0),
                "elastic_5.0": linear_model.ElasticNet(alpha=5.0),
                "elastic_10.0": linear_model.ElasticNet(alpha=10.0),
                "elastic_15": linear_model.ElasticNet(alpha=15),
                "ridge_0.01": linear_model.Ridge(alpha=0.01),
                "ridge_0.1": linear_model.Ridge(alpha=0.1),
                "ridge_1.0": linear_model.Ridge(alpha=1.0),
                "ridge_5.0": linear_model.Ridge(alpha=5.0),
                "ridge_10.0": linear_model.Ridge(alpha=10.0),
                "ridge_15": linear_model.Ridge(alpha=15),
                "lasso_0.01": linear_model.Lasso(alpha=0.01),
                "lasso_0.1": linear_model.Lasso(alpha=0.1),
                "lasso_1.0": linear_model.Lasso(alpha=1.0),
                "lasso_5.0": linear_model.Lasso(alpha=5.0),
                "lasso_10.0": linear_model.Lasso(alpha=10.0),
                "lasso_15": linear_model.Lasso(alpha=15),
                "knn_5": neighbors.KNeighborsRegressor(n_neighbors=5),
                "knn_10": neighbors.KNeighborsRegressor(n_neighbors=10),
                "knn_15": neighbors.KNeighborsRegressor(n_neighbors=15),
                "knn_20": neighbors.KNeighborsRegressor(n_neighbors=20),
                "knn_25": neighbors.KNeighborsRegressor(n_neighbors=25),
                "knn_30": neighbors.KNeighborsRegressor(n_neighbors=30),
                }       
            
            #select j random estimators
            casuale = random.sample(list(library1.keys()), lib_size[j])
            lib = {k: library1[k] for k in casuale}
            
            
            # Initialize the learner
            if type == "sequential":
                learner = sl.SuperLearner(lib)
            if type == "parallel":
                learner = psl.SuperLearner(lib)
            
            # Generate dataset
            beta = np.random.randint(low = -20, high = 20, size = 10)
            X, y = ms.make_regression_fixed_coeffs(n_samples = sample[i], n_features = 10, coefficients = beta, noise = 10)
            
            start_time = time.time()
            learner.fit(X, y)
            end_time = time.time()
            
            times[i, j] = end_time - start_time
            
            library1.clear()
    
    return times
    
def speedUp():
    
    sample = [100, 500, 1000, 5000, 7500, 10000]
    lib_size = [5, 10, 15, 20, 24]
    
    times_seq = np.zeros((len(sample), len(lib_size)))
    times_par = np.zeros((len(sample), len(lib_size)))
    
    for i in tqdm.tqdm(range(5)):
        times_seq =+ timeTest("sequential", sample, lib_size)

    times_seq = times_seq / 5
    
    for j in tqdm.tqdm(range(5)):
        times_par =+ timeTest("parallel", sample, lib_size)
        
    times_par = times_par / 5
    
    np.savetxt("times_seq.csv", times_seq, delimiter=",")
    np.savetxt("times_par.csv", times_par, delimiter=",")
    
                    
if __name__ == "__main__":
    montecarloLibrary()