import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import parallelSuperLearner as psl

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from imblearn.over_sampling import SMOTE
from collections import Counter

def main():
    path = os.getcwd()
    df = pd.read_csv(path + "/Data/diabetes_prediction_dataset.csv")
    print(df.head(), "\n")
    print(df.info(), "\n")
    
    # Handle duplicates
    duplicate_rows_data = df[df.duplicated()]
    print("number of duplicate rows: ", duplicate_rows_data.shape, "\n")
    df = df.drop_duplicates()
    
    # Loop through each column and count the number of distinct values
    for column in df.columns:
        num_distinct_values = len(df[column].unique())
        print(f"{column}: {num_distinct_values} distinct values")
    
    # Checking null values
    print(df.isnull().sum())
        
    # Remove Unneccessary value [0.00195%]
    df = df[df['gender'] != 'Other']
    
    df_sample = df.sample(n=10000, random_state=1)
    # Count plots for binary variables
    sns.pairplot(df_sample, hue='diabetes')
    plt.show()
    # Define a function to map the existing categories to new ones
    
    #df['gender'] = df['gender'].replace(['Female'], '1')
    #df['gender'] = df['gender'].replace(['Male'], '0')
    #data = pd.get_dummies(df, columns=['smoking_history'])
    #
    #scaler = StandardScaler()
    #df["age"] = scaler.fit_transform(df[["age"]])
    #df["bmi"] = scaler.fit_transform(df[["bmi"]])
    #df["blood_glucose_level"] = scaler.fit_transform(df[["blood_glucose_level"]])
    #df["HbA1c_level"] = scaler.fit_transform(df[["HbA1c_level"]])
    #
    #y = data.pop('diabetes') 
    #
    #smote = SMOTE(k_neighbors=20, n_jobs=-1)
    #x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.5)
    #x_train, y_train = smote.fit_resample(x_train, y_train)
    #
    #print("Original dataset shape %s" % Counter(y))
    #print("Resampled dataset shape %s" % Counter(y_train))
    #print(len(x_train), len(y_train))
    #
    #library = {'RandomForest': RandomForestClassifier(n_estimators=20),
    #           "GradientBoosting": GradientBoostingClassifier(n_estimators=20),
    #           "AdaBoost2": AdaBoostClassifier(n_estimators=25),
    #           "LogisticRegression": LogisticRegression(),
    #           "KNeighbors_10": KNeighborsClassifier(n_neighbors=10),
    #           "KNeighbors_20": KNeighborsClassifier(n_neighbors=20),
    #           "KNeighbors_50": KNeighborsClassifier(n_neighbors=50),
    #           "KNeighbors_100": KNeighborsClassifier(n_neighbors=100),
    #           "KNeighbors_200": KNeighborsClassifier(n_neighbors=200),
    #}
    #
    #library2 = {'RandomForest': RandomForestClassifier(n_estimators=10),
    #            "RandomForest2": RandomForestClassifier(n_estimators=25),
    #            "RandomForest3": RandomForestClassifier(n_estimators=50),
    #            "RandomForest4": RandomForestClassifier(n_estimators=100),
    #            "GradientBoosting": GradientBoostingClassifier(n_estimators=10),
    #            "GradientBoosting2": GradientBoostingClassifier(n_estimators=25),
    #            "GradientBoosting3": GradientBoostingClassifier(n_estimators=50),
    #            "GradientBoosting4": GradientBoostingClassifier(n_estimators=100),
    #            "AdaBoost": AdaBoostClassifier(n_estimators=10),
    #            "AdaBoost2": AdaBoostClassifier(n_estimators=25),
    #            "AdaBoost3": AdaBoostClassifier(n_estimators=50),
    #            "AdaBoost4": AdaBoostClassifier(n_estimators=100),
    #            "LogisticRegression": LogisticRegression(),
    #            "KNeighbors_10": KNeighborsClassifier(n_neighbors=10),
    #            "KNeighbors_20": KNeighborsClassifier(n_neighbors=20),
    #            "KNeighbors_50": KNeighborsClassifier(n_neighbors=50),
    #            "KNeighbors_100": KNeighborsClassifier(n_neighbors=100),
    #            "KNeighbors_200": KNeighborsClassifier(n_neighbors=200),
    #}
    #    
    #sl = psl.SuperLearner(library2, task='regression', verbose=False, folds = 5)
    #sl.fit(x_train, y_train)
    #
    #for name, model in library2.items():
    #    model.fit(x_train, y_train)
    #    
    #y_pred = np.array(sl.predict(x_test))
    #y_pred = np.round(y_pred)
    #y_pred = y_pred.astype(int)
    #weights = sl.weights
    #
    #print("Super Learner Accuracy: ", "\n",  accuracy_score(y_test, y_pred))
    #print("Super Learner Classification Report: ", "\n", classification_report(y_test, y_pred))
    #print("Super Learner Confusion Matrix: ", "\n", confusion_matrix(y_test, y_pred))
    #print("Super Learner weights: ", "\n", weights)
    #
    #for i in range(len(weights)):
    #    if weights[i] > 0:
    #        print("Model: ", list(library2.keys())[i], "Weight: ", weights[i])
    #        print("Model Accuracy: ", accuracy_score(y_test, library2[list(library2.keys())[i]].predict(x_test)))
    #        print("Model risk: ", np.mean((y_test - library2[list(library2.keys())[i]].predict(x_test))**2))
    #        print(" ")
#
    #
main()