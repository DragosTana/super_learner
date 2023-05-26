import pandas as pd
import numpy as np
import misc as ms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from matplotlib.cm import ScalarMappable

def dataVisualization(estimators, sample_sizes):

    num_models = len(estimators)
    
    mean_mses = np.zeros((num_models, len(sample_sizes)))
    for i, sample_size in enumerate(sample_sizes):
        mse_df = pd.read_csv("Data/R2_at_{}.csv".format(sample_size))
        mean_mses[:, i] = mse_df.mean()
        
    # Plotting the line plot
    for model_idx in range(num_models):
        plt.plot(sample_sizes, mean_mses[model_idx, :],linewidth = 3, label='{}'.format(estimators[model_idx]))

    plt.xlabel('Sample Size')
    plt.ylabel("R2")
    plt.title('Performance Trend: R2 vs Sample Size')
    plt.legend()
    plt.show()

def speedUpVisual():
    times_psl = np.loadtxt("times_par.csv", delimiter=",")
    times_sl = np.loadtxt("times_seq.csv", delimiter=",")
    
    speeup = times_sl / times_psl
    
    sample = [100, 500, 1000, 5000, 7500, 10000, 15000, 20000]
    lib_size = [5, 10, 15, 20, 24]
    
    
    
    plt.plot(sample, speeup[:, 1], linewidth = 3, label='{}'.format(lib_size[1]))
    plt.plot(sample, speeup[:, 4], linewidth = 3, label='{}'.format(lib_size[4]))
    plt.xlabel('Sample Size')
    plt.ylabel("SpeedUp")
    plt.title('Performance Trend: SpeedUp vs Sample Size')
    plt.legend()
    plt.show()
    
    
def weightsVisual(path):
    df = pd.read_csv(path)
    averages = df.mean()
    std_devs = df.std()
    conf_ints = 1.96 * std_devs / np.sqrt(len(df))  
    plt.bar(averages.index, averages.values, yerr=conf_ints, capsize=4, color='green')

    plt.title('Average Values with Confidence Intervals')
    plt.xlabel('Base Estimators')
    plt.ylabel('Average Values')
    plt.show()
    

if __name__ == "__main__":
    path = "Data/weights_opt_frid.csv"
    weightsVisual(path)


    
    
    