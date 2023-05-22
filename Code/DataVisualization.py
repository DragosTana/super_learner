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
    times_psl = np.loadtxt("times_psl.csv", delimiter=",")
    times_sl = np.loadtxt("times_sl.csv", delimiter=",")
    
    sample = [100, 200, 500, 1000, 2000, 5000, 7500, 10000]
    lib_size = [5, 10, 20, 50, 80, 100]
    
    X, Y = np.meshgrid(sample, lib_size)
    
    z = times_sl / times_psl
    
    X = X.flatten()
    Y = Y.flatten()
    Z = np.array(z).flatten()
    
    # Create triangulation
    tri = Triangulation(X, Y)

    # Create figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_trisurf(X, Y, Z, triangles=tri.triangles, cmap="viridis", edgecolor="none")
    sm = ScalarMappable(cmap='viridis')
    sm.set_array(Z)
    fig.colorbar(sm)
    ax.legend()
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Library Size')
    ax.set_zlabel('Speed Up')
    ax.set_title('Speed Up vs Sample Size and Library Size')

    # Show the plot
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
    path = "Data/sl_scores.csv"
    weightsVisual(path)


    
    
    