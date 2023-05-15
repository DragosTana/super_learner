import pandas as pd
import numpy as np
import misc as ms
import matplotlib.pyplot as plt

def dataVisualization(estimators, sample_sizes):

    num_models = len(estimators)
    
    mean_mses = np.zeros((num_models, len(sample_sizes)))
    for i, sample_size in enumerate(sample_sizes):
        mse_df = pd.read_csv("Data/R2_at_{}.csv".format(sample_size))
        mean_mses[:, i] = mse_df.mean()
        
    # Plotting the line plot
    for model_idx in range(num_models):
        plt.plot(sample_sizes, mean_mses[model_idx, :],linewidth = 3, label='Model {}'.format(estimators[model_idx]))

    plt.xlabel('Sample Size')
    plt.ylabel("R2")
    plt.title('Performance Trend: R2 vs Sample Size')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("Data/R2_at_100.csv")
    estimators = df.columns
    sample_sizes = [100, 200, 500, 1000, 5000, 10000]
    dataVisualization(estimators, sample_sizes)