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

import misc as ms
import parallelSuperLearner as psl
import superLearner as sl

def main():
    pass
    
def montecarloLibrary():
    
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
    
    sample_sizes = [100, 200, 500, 1000]
    
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
    
def montecarloFoldNumber():
    pass
    
    
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
    sim = 100
    
    for i in tqdm.tqdm(range(sim)):
        #X, y = datasets.make_friedman1(1000, noise=0.5)
        X, y = datasets.make_regression(n_samples = 1000, n_features = 20, n_informative=10,  noise = 30)    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
        y_test = scaler.fit_transform(y_test.reshape(-1,1)).flatten()
        
        sl1 = sl.SuperLearner(library, meta_learner=linear_model.ElasticNetCV(alphas=np.arange(0.05, 15.0, 0.05)))
        sl2 = sl.SuperLearner(library)
        
        sl1.fit(X_train, y_train)
        sl2.fit(X_train, y_train)
        
        #iteration_score = []
        #for estimator in library:
        #    library[estimator].fit(X_train, y_train)
        #    s = library[estimator].score(X_test, y_test)
        #    iteration_score.append(s)
        #
        #weigths_meta.append(sl1.weights)
        #weigths_opt.append(sl2.weights)
        #scores.append(iteration_score)    
        sl_scores.append([sl1.score(X_test, y_test), sl2.score(X_test, y_test)])
        
    #weigths_meta = np.array(weigths_meta)
    #weigths_opt = np.array(weigths_opt)
    #scores = np.array(scores)
    sl_scores = np.array(sl_scores)
    
    #pd_meta = pd.DataFrame(weigths_meta, columns=list(library.keys()))
    #pd_opt = pd.DataFrame(weigths_opt, columns=list(library.keys()))
    #pd_score = pd.DataFrame(scores, columns=list(library.keys()))
    pd_sl_score = pd.DataFrame(sl_scores, columns=["meta", "opt"])
    
    #pd_meta.to_csv(path + "/Data/weights_meta.csv", index=False)
    #pd_opt.to_csv(path + "/Data/weights_opt.csv", index=False)
    #pd_score.to_csv(path + "/Data/scores.csv", index=False)
    pd_sl_score.to_csv(path + "/Data/sl_scores.csv", index=False)
    
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
    montecarloOptimization()
