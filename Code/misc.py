

import numpy as np

def make_regression_fixed_coeffs(n_samples, n_features, coefficients, noise=0.0, random_state=None):
    """
    Generate a regression dataset with fixed coefficients.

    Parameters:
        n_samples (int): The number of samples.
        n_features (int): The number of features.
        coefficients (array-like): The fixed coefficients to be used.
        noise (float, optional): The standard deviation of the Gaussian noise added to the output.
        random_state (int, optional): Seed for the random number generator.

    Returns:
        X (ndarray): The input features.
        y (ndarray): The target values.
    """
    if n_features != len(coefficients):
        raise ValueError("The number of features must be equal to the number of coefficients.")
    
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    y = np.dot(X, coefficients) + noise * rng.standard_normal(n_samples)
    return X, y