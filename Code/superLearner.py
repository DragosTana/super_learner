from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import numpy as np

class SuperLearner(BaseEstimator, RegressorMixin):
    
    def __init__(self, base_estimators):
        self.base_estimators = base_estimators
        self.meta_learner = LinearRegression(positive=True)
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
                
        print(meta_predictions, y)
        
        self.weights = self.meta_learner.fit(meta_predictions, y).coef_
        self.weights = self.weights / np.sum(self.weights)
        
        return self
    
    def predict(self, X):
        check_is_fitted(self, 'meta_learner')
        X = check_array(X)
        
        base_predictions = np.zeros((X.shape[0], len(self.base_estimators)), dtype=np.float64)
        for i, estimator in enumerate(self.base_estimators):
            base_predictions[:, i] = estimator.predict(X)
            
        return np.dot(base_predictions, self.weights)
    