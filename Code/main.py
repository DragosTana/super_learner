from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import superLearner

# Load dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base estimators
estimators = [
    make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42)),
    LinearRegression()
]

# Create SuperLearner estimator
super_learner = SuperLearner(base_estimators=estimators)

# Train the SuperLearner estimator
super_learner.fit(X_train, y_train)

# Make predictions on the test set
predictions = super_learner.predict(X_test)

# Evaluate the performance
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
