import pandas as pd
import numpy as np
import misc as ms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from matplotlib.cm import ScalarMappable
import seaborn as sns

def dataVisualization(estimators, sample_sizes):

    num_models = len(estimators)
    
    mean_mses = np.zeros((num_models, len(sample_sizes)))
    for i, sample_size in enumerate(sample_sizes):
        mse_df = pd.read_csv("Data/R2_at_{}_lib.csv".format(sample_size))
        mean_mses[:, i] = mse_df.mean()
        
    # Plotting the line plot
    for model_idx in range(num_models):
        plt.plot(sample_sizes, mean_mses[model_idx, :],linewidth = 3, label='{}'.format(estimators[model_idx]))

    plt.xlabel('Sample Size')
    plt.ylabel("R2")
    plt.title('Performance Trend: R2 vs Sample Size')
    plt.legend()
    plt.show()
    
def boxPlotR2(ml_techniques, sample_sizes):
    dfs = []

    for sample_size in sample_sizes:
        file_name = f"Data/R2_at_{sample_size}_folds.csv"
        df = pd.read_csv(file_name)
        df['Sample Size'] = sample_size
        dfs.append(df)

    combined_df = pd.concat(dfs)
    combined_df = pd.melt(combined_df, id_vars=['Sample Size'], var_name='ML Technique', value_name='R2 Score')

    # Find the ML technique with the highest R2 score for each sample size
    highest_scores = combined_df.groupby('Sample Size')['R2 Score'].transform('max')
    combined_df['Highest Score'] = combined_df['R2 Score'] == highest_scores

    palette = sns.color_palette('Set2', n_colors=len(ml_techniques))
    palette[-1] = 'red'  # Set the last color in the palette to red

    ax = sns.boxplot(data=combined_df, x='Sample Size', y='R2 Score', hue='ML Technique', showfliers=False, palette=palette)

    # Color the boxes with the highest scores in red
    for patch in ax.artists:
        # Find the ML technique with the highest score for each patch (box)
        highest_technique = combined_df.loc[combined_df['ML Technique'] == patch.get_label(), 'Highest Score'].any()
        if highest_technique:
            patch.set_facecolor('red')

    plt.xlabel('Sample Size')
    plt.ylabel('R2 Score')
    plt.title('Grouped Boxplot of R2 Scores')

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
    plt.yscale('log')
    plt.legend()
    plt.show()
    
def weightsVisual(path):
    df = pd.read_csv(path)
    averages = df.mean()
    print(averages)
    std_devs = df.std()
    conf_ints = 1.96 * std_devs / np.sqrt(len(df))  
    col = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    plt.bar(averages.index, averages.values, yerr=conf_ints, capsize=4, color = col)

    plt.title('Average Values with Confidence Intervals')
    plt.xlabel('Base Estimators')
    plt.ylabel('Average Values')
    plt.show()

if __name__ == "__main__":
    sample_sizes = [100, 200, 500, 1000, 5000, 10000, 15000]
    columns=["5-fold", "10-fold", "Adaptive"]
    boxPlotR2(columns, sample_sizes)
    
    
    
    