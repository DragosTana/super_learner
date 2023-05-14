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
import superLearner as sl

def main():
    pass
    
def montecarlo():
    
    sim = 20
    n = 100
    beta = np.ones(5)
    
    for i in range(sim):
        np.random.seed(2)
        X, y = ms.make_regression_fixed_coeffs(n_samples = 10, n_features = len(beta), coefficients = beta, noise = 1.0)
        print(X)
        print(y)
        
    
if __name__ == "__main__":
    montecarlo()