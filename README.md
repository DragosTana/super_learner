# Super Learners

This repository contains sklearn compatible superLearner estimator. Both sequential and parallel version of the estimator was implemented gaining a speed up of 7x in the parallel one. The superLearner class can be used both for classification and regression tasks.
For regression tasks the weights of each base estimator can be calculated solving the optimization problem that minimizes the sum of squares between the meta predictions and the target variable or passing a meta_learner (i.e. any parametric estimator of sklearn framework will be fine).
For the classification task the weights are calculated using the accuracy of each estimator. Different metrics will be implemented.

## Speedup of the parallel version

![speedup](/Images/SpeedUp.png "speedup")

## Usage

Define a library of learners as a dictionary as follows:

``
library = {
    "ols": linear_model.LinearRegression(),
    "ridge": linear_model.RidgeCV(alphas=np.arange(0.05, 15, 0.05)),
    "lasso" : linear_model.LassoCV(alphas=np.arange(0.05, 15.0, 0.05)),
    "elastic" :  linear_model.ElasticNetCV(alphas=np.arange(0.05, 15.0, 0.05)),
    "knn_5": neighbors.KNeighborsRegressor(n_neighbors=5),
    "knn_10": neighbors.KNeighborsRegressor(n_neighbors=10),
    "knn_15": neighbors.KNeighborsRegressor(n_neighbors=15),
}
``
The Super Learner class is sklearn compatible estimator. As such it has the typical fit and predicts methods.
To instanciate the Super Learner we can proceed as follows:

``
sl = SuperLearner(base_estimators = library, folds = 10,  meta_learner = None, task = 'regression', threshold = 0.01, verbose = False)

``