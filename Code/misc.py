from scipy import stats
import numpy as np
import csv

def load_csv(file_name):
    data = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            point = []
            for word in line:
                point.append(float(word))
            data.append(point)
    return np.array(data)

def calculate_mean_ci(data):
    n = len(data)
    mean = np.mean(data)
    std_error = stats.sem(data)
    confidence_interval = std_error * stats.t.ppf((1 + 0.95) / 2, n - 1)
    return mean, confidence_interval

def make_regression_fixed_coeffs(n_samples, n_features, coefficients, noise=0.0, random_state=None):
    """
    Generate a regression dataset with fixed coefficients.

    ## Parameters:
        n_samples: int 
            The number of samples.
            
        n_features: int
            The number of features.
            
        coefficients: ndarray
            The coefficients of the linear model.
            
        noise : float, default=0.0
            The standard deviation of the gaussian noise applied to the output.
            
        random_state : int, RandomState instance or None, default=None
            Determines random number generation for dataset creation. Pass an int for reproducible output across multiple function calls. See Glossary <random_state>.

    ## Returns:
    X : ndarray of shape (n_samples, n_features)
        The input samples.

    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        The output values.
        
    """
    if n_features != len(coefficients):
        raise ValueError("The number of features must be equal to the number of coefficients.")
    
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    y = np.dot(X, coefficients) + noise * rng.standard_normal(n_samples)
    return X, y