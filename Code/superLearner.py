from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import neighbors
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy import optimize
from pandas.plotting import scatter_matrix
import numpy as np
import pandas as pd 

class SuperLearner(BaseEstimator, RegressorMixin):
    
    def __init__(self, base_estimators, meta_learner = None):
        self.base_estimators = base_estimators.values()
        self.meta_learner = meta_learner
        self.weights = None
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        
        meta_predictions = np.zeros((X.shape[0], len(self.base_estimators)), dtype=np.float64)
        #TODO: modify the number of folds depending on the number of base estimators and the size of the dataset
        kf = KFold(n_splits=5)        
        
        for i, (tran_idx, val_idx) in enumerate(kf.split(X)):
            
            X_train, X_val = X[tran_idx], X[val_idx]
            y_train, y_val = y[tran_idx], y[val_idx]
            for j, estimator in enumerate(self.base_estimators):
                estimator.fit(X_train, y_train)
                meta_predictions[val_idx, j] = estimator.predict(X_val)
                
        #df = pd.DataFrame(np.hstack((meta_predictions, y.reshape(-1,1))))
        #last_column_index = df.shape[1] - 1
        #df.rename(columns={last_column_index: 'y'}, inplace=True)
        #names = {i : estimator.__class__.__name__ for i, estimator in enumerate(self.base_estimators)}
        #df.rename(columns=names, inplace=True)
        #print(df.head())
        #print(" ")
        
        if self.meta_learner is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(meta_predictions)
            y_scaled = scaler.fit_transform(y.reshape(-1,1)).flatten()
            result = optimize.nnls(X_scaled, y_scaled)
            result = result[0]
            result = result / np.sum(result)
            self.weights = result
            print(result, np.sum(result))
            print(" ")
        else :
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(meta_predictions)
            y_scaled = scaler.fit_transform(y.reshape(-1,1)).flatten()
            self.meta_learner.fit(X_scaled, y_scaled)
            result = self.meta_learner.coef_
            result = result / np.sum(result)
            self.weights = result
            print(result, np.sum(result))
            print(" ")
        
        for estimator in self.base_estimators:
            estimator.fit(X, y)
        
        return self
    
    def predict(self, X):
        check_is_fitted(self, 'meta_learner')
        X = check_array(X)
        
        base_predictions = np.zeros((X.shape[0], len(self.base_estimators)), dtype=np.float64)
        for i, estimator in enumerate(self.base_estimators):
            base_predictions[:, i] = estimator.predict(X)
            
        return np.dot(base_predictions, self.weights)
   
###########################################################################################################################à    

def generate_data(features = 10, n = 1000, plot = False):
    
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

    return X, Y

def main():
    
    X, y = datasets.make_friedman1(1000)
    #X, y = generate_data(features = 50, n = 600, plot = False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    library = {
        "ols": linear_model.LinearRegression(),
        "elastic": linear_model.ElasticNetCV(n_alphas=200, l1_ratio=0.5),
        "ridge": linear_model.RidgeCV(alphas=np.arange(0.1, 10, 0.1)),
        "lars": linear_model.LarsCV(),
        "lasso": linear_model.LassoCV(alphas=np.arange(0.1, 10, 0.1)),
        "knn": neighbors.KNeighborsRegressor()
    }
    
    superLeaner1 = SuperLearner(library)
    
    superLeaner1.fit(X_train, y_train)
    y_pred = superLeaner1.predict(X_test)
    
    superLeaner2 = SuperLearner(library, linear_model.LinearRegression(positive=True))
    superLeaner2.fit(X_train, y_train)
    
    print(" ")
    print("R^2 without meta learner: ", superLeaner1.score(X_test, y_test))
    print("R^2 with meta learner: ", superLeaner2.score(X_test, y_test))
    print(" ")
    
    for i, estimator in enumerate(superLeaner1.base_estimators):
        print("R^2 for {name}: {score}".format(name = estimator.__class__.__name__, score = estimator.score(X_test, y_test)))
    
if __name__ == "__main__":
    main()
    
