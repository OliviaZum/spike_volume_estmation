import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from icecream import ic
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    
def compare(mapping_true : pd.DataFrame, 
            mapping_predicted : pd.DataFrame, 
            output_name : str,
            name : str,
            output_folder : Path = None,
            verbose : bool = True):
    """Compute and print statistics

    Args:
        mapping_true (pd.DataFrame): Mapping file with true / measured volumes 
        mapping_predicted (pd.DataFrame): Estimated volumes
        output_name (str): Volume column name
        name (str): Test, Train, or Val
        output_folder (str): Path to output folder
        verbose (bool) 
    """
    
    n_experiments = mapping_predicted.shape[0]

    # Check for valid plant_id matches
    mapping_predicted = mapping_predicted.loc[mapping_predicted['plant_id'].isin(mapping_true.index)]
   
    true_values = np.array(mapping_true.loc[mapping_predicted["plant_id"], output_name])
    pred_values = np.array(mapping_predicted[output_name])

    if verbose:
        print(f"Stats for estimations")
        print(f"Number of compared experiments {mapping_predicted.shape[0]}")
        print(f"Number of total experiments {n_experiments}")

    diff_volumes = np.abs(pred_values - true_values)
    rel_diff_volumes = diff_volumes / true_values
    
    # Create a DataFrame
    save = pd.DataFrame({'True': true_values, 'Predicted': pred_values})
    save.to_csv(f'{output_folder}/{name}_vs_true.csv', index=False)
  
    MAE = mean_absolute_error(true_values, pred_values)
    MSE = mean_squared_error(true_values, pred_values)
    RMSE = np.sqrt(MSE)

    #MAPE:
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0  # Avoid division by zero
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    MAPE = mean_absolute_percentage_error(true_values, pred_values)
    r_squared = r2_score(true_values, pred_values)
    correlation = np.corrcoef(true_values, pred_values)
    cor = correlation[0,1]
    
    #Dictionary with statistics:
    statistics = pd.DataFrame(columns=["metric", "value"])
    
    statistics.loc[0] = {"metric": "mean_error", "value": np.mean(diff_volumes)}
    statistics.loc[1] = {"metric": "max_error", "value": np.max(diff_volumes)}
    statistics.loc[2] = {"metric": "min_error", "value": np.min(diff_volumes)}
    statistics.loc[3] = {"metric": "std_error", "value": np.std(diff_volumes)}
    statistics.loc[4] = {"metric": "rel_mean_error", "value": np.mean(rel_diff_volumes)}
    statistics.loc[5] = {"metric": "rel_max_error", "value": np.max(rel_diff_volumes)}
    statistics.loc[6] = {"metric": "rel_min_error", "value": np.min(rel_diff_volumes)}
    statistics.loc[7] = {"metric": "rel_std_error", "value": np.std(rel_diff_volumes)}
    statistics.loc[8] = {"metric": "MAE", "value": MAE}
    statistics.loc[9] = {"metric": "MSE", "value": MSE}
    statistics.loc[10] = {"metric": "RMSE", "value": RMSE}
    statistics.loc[11] = {"metric": "MAPE", "value": MAPE} #“How much (what %) of the total variation in Y(target) is explained by the variation in X(regression line)”
    statistics.loc[12] = {"metric": "R2", "value": r_squared}
    statistics.loc[13] = {"metric": "Correlation", "value": cor}
    
    # Save the DataFrame to a CSV file with the specified path and variable name
    statistics.to_csv(f'{output_folder}/{name}_statistics.csv', index=False)

    # Plotting the scatter plot
    fig, ax = plt.subplots()
    ax.scatter(true_values, pred_values, color='black', label='True vs Predicted', marker='o', s=4)
    ax.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], linestyle='--', color='red', label='Perfect Predictions')
    ax.set_xlabel('True Volumes [mm³]')
    ax.set_ylabel('Predicted Volumes [mm³]')
    ax.text(0.05, 0.9, f'Corr: {cor:.2f}', transform=plt.gca().transAxes, color='black')
    ax.set_facecolor("white")
    fig.set_facecolor("white")
    
    #name to save
    file_name = f'{output_folder}/scatter_plot_{name}.png'

    # Save the plot to a file (e.g., PNG, PDF, etc.)
    fig.savefig(file_name)
    fig.tight_layout()
    
    if verbose:
        print(statistics)
        
    return fig, statistics