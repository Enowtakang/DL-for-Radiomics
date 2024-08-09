import os.path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


"""
Set paths to dataset and results
"""
path_to_dataset = "C:/Users/HP/PycharmProjects/MPH/4results/1_numerical_evaluations/numerical_results.csv"
path_to_results = "C:/Users/HP/PycharmProjects/MPH/4results/1_numerical_evaluations/analysis_of_numerical_results/"

"""
Load dataset, 
        - drop the 'Model' column and 
        - prepare a list of variables
"""
data = pd.read_csv(path_to_dataset)
data = data.drop(columns=['Model', 'Recall'])
variables = data.columns

"""
Compute sensitivity. 
Procedure:
            - Define a function to compute sensitivity
            - Initialize a DataFrame to store sensitivity results
            - Perform and visualize pairwise comparisons
            - Save results to CSV
"""


def compute_sensitivity(x, y, epsilon=1e-10):
    dx = np.diff(x)
    dy = np.diff(y)
    sensitivity = np.abs(dy / (dx + epsilon))
    return sensitivity


def sensitivity_analysis():
    # Initialize a DataFrame to store sensitivity results
    sensitivity_results = pd.DataFrame(
        columns=['Variable1', 'Variable2', 'Sensitivity'])

    # Perform and visualize pairwise comparisons
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            var1 = variables[i]
            var2 = variables[j]

            # Scatter plot
            plt.figure(figsize=(10, 10))
            sns.scatterplot(x=data[var1], y=data[var2])
            plt.title(f'Scatter Plot: {var1} vs {var2}')
            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.savefig(os.path.join(
                path_to_results,
                f'scatter_plot_of_{str.upper(var1)}_vs_{str.upper(var2)}.png'))
            plt.close()

            # Sensitivity analysis
            sensitivity = compute_sensitivity(data[var1].values, data[var2].values)
            avg_sensitivity = np.mean(sensitivity)
            sensitivity_results = pd.concat(
                [sensitivity_results, pd.DataFrame({
                    'Variable1': [var1], 'Variable2': [var2],
                    'Sensitivity': [avg_sensitivity]})], ignore_index=True)

    # Save sensitivity results to CSV
    sensitivity_results.to_csv(os.path.join(
        path_to_results,
        '_sensitivity_results.csv'), index=False)


sensitivity_analysis()


"""
Save and visualize correlation matrix
"""


def correlation_matrix():
    # Save correlation matrix
    correlation_matrix = data.corr()
    correlation_matrix.to_csv(os.path.join(
        path_to_results,
        '_correlation_matrix.csv'))

    # Heatmap of correlation matrix
    plt.figure(figsize=(12, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix Heatmap')
    plt.savefig(os.path.join(
        path_to_results,
        '_correlation_matrix_heatmap.png'))
    plt.close()


correlation_matrix()
