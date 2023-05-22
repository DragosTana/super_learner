#Esperimento con R 
library(SuperLearner)
n <- 5000
p <- 5

#y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 
# 10 * X[:, 3] + 5 * X[:, 4] + noise * N(0, 1).


X <- matrix(rnorm(n*p,0, 1), ncol = p)
Y <- 10*sin(pi*X[, 1] * X[, 2]) +  20 * (X[, 3] - 0.5)^2 + 10 * X[, 4] +
  + 5*X[,5] + rnorm(n)

train_obs <- sample(n,ceiling(0.8*n))
x_train <- X[train_obs, ]
x_holdout <- X[-train_obs, ]
y_train <- Y[train_obs]
y_holdout <- Y[-train_obs]
x_train <- as.data.frame(x_train)


#Costruisco il superlearner
listWrappers() #Restituisce i learner e i metodi di screening a disposizione
#"SL.glm","SL.ranger","SL.xgboost","SL.knn","SL.glmnet" 

set.seed(123)
"
SL.library_1 <- list(
  SL.lm = list("library" = "stats"),
  SL.glmnet_1 = list("library" = "glmnet", "alpha" = 1, "lambda" = c(0.1, 1, 10)),
  SL.glmnet_2 = list("library" = "glmnet", "alpha" = 0.5, "lambda" = c(0.1, 1, 10)),  # Elastic Net
  SL.glmnet_3 = list("library" = "glmnet", "alpha" = 0, "lambda" = c(0.1, 1, 10)),  # Ridge
  SL.knn_15 = list("library" = "class", "k" = 15),  # KNN con k = 15
  SL.gboost_1 = list("library" = "gbm", "n.trees" = 50),  # Gradient Boosting Machine (gboost)
  SL.ranger_1 = list("library" = "ranger", "num.trees" = 50)  # Random Forest (sl.ranger)
)
learner <- c("SL.lm", "SL.glmnet1", "SL.glmnet2", "SL.glmnet3", "SL.knn15", "SL.gboost1", "SL.ranger1")
SL.glmnet1 = list("library" = "glmnet", "alpha" = 1, "lambda" = c(0.1, 1, 10))
"#NON FUNZIONA

SL.rf.1 = function(...) {
  SL.randomForest(..., num.trees = 50)
}

SL.gbm.1 = function(...) {
  SL.gbm(..., gbm.trees = 50)
}

SL.KNN.1 = function(...) {
  SL.knn(..., k = 15)
}

l <- c(0.1,1,10)

SL.glmnet_1 = function(...){
  SL.glmnet(...,alpha = 0)
}
learnersRidge = create.Learner("SL.glmnet_1", tune = list(lambda = l))

SL.ridge_1 = function(...){
  SL.glmnet_1(...,lambda = c(0.1,1,10))
}

SL.glmnet_2 = function(...){
  SL.glmnet(...,alpha = 0.5)
}
learnersEN = create.Learner("SL.glmnet_2", tune = list(lambda = l))

sl = SuperLearner(Y = y_train, X = x_train, family = gaussian(),
                  SL.library = c("SL.rf.1","SL.glm",
                                  learnersEN$names))
sl

str(sl_lasso$fitLibrary$SL.glmnet_All$object, max.level = 1)







SL.glmnet_2 = function(...){
  glmnet(...,alpha = 0.5)
}
learnersEN = create.Learner("SL.glmnet_2", tune = list(lambda = l))