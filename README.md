# Super Learners

This repository contains sklearn compatible superLearner estimator. Both sequential and parallel version of the estimator was implemented gaining a speed up of 2x in the parallel one. The superLearner class can be used both for classification and regression tasks.
For regression tasks the weights of each base estimator can be calculated solving the optimization problem that minimizes the sum of squares between the meta predictions and the target variable or passing a meta_learner (i.e. any parametric estimator of sklearn framework will be fine).
For the classification task the weights are calculated using the accuracy of each estimator. Different metrics will be implemented.
