import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error , mean_absolute_error
import numpy as np
import matplotlib.patches as mpatches
import json
from sklearn.linear_model import LinearRegression
import click
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

def metrics_baseline(output_folder_area: str, 
                  input_folder_geometric: str,
                  output_folder_geometric: str,
                  mapping_train: str, 
                  number_of_images: int, 
                  baseline_geometric: bool): 
    
    """
    Calculates the correlation, R^2, and MAPE for true volumes vs scaled predicted volumes. 
    Scaling factor is being calculated from the training data set. Saves a correlation plot
    in output folder. 
    
    
    Args: 
        output_folder_area (str): Path to output folder of the script area_baseline.py. 
        input_folder_geometric (str): Path to folder where predicted volumes of geometric baseline is saved (volume_basleine.csv)
        output_folder_geometric (str): Path to folder where output of geometric baseline will be saved. Will be created if it does not exist.
        mapping_train (str): Path to train mapping file
        number_of_images (int): Number of images per spike considered in calculation of metrics
        baseline_geometric (bool): if True, metrics for geometric baseline will be calculated, if False
        metrics for area baseline will be calculated
    """

    # Function to extract the first three parts of the img_id
    def get_plant_id(img_id):
        parts = img_id.split('_')
        plant_id = '_'.join(parts[:3])
        return plant_id
    
    # Function to calcualte MAPE
    def mean_absolute_percentage_error(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            mask = y_true != 0  # Avoid division by zero
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    ########## ground truth training data #############
    #load training spikes 

    # Load the JSON training file
    with open(mapping_train, 'r') as f:
        data = json.load(f)
        
    # Initialize empty lists to store the data
    plant_ids = []
    volumes = []

    # Iterate through each entry in the JSON data
    for plant_id, info in data.items():
        # Extract the 'volume' information
        volume = info['volume']
        
        # Append the 'Plant_id' and 'volume' to the lists
        plant_ids.append(plant_id)
        volumes.append(volume)

    # Create a dataframe from the lists
    df_train_true = pd.DataFrame({'Plant_id': plant_ids, 'Volume_train': volumes})


    #########################################################################

    #Load
    if baseline_geometric:
        path_volume_true = output_folder_area / 'volume_test.csv' #measured volume of test set
        path_volume_pred = input_folder_geometric / 'volume_baseline.csv' #predicted volume 
        df_true = pd.read_csv(path_volume_true) 
        df_true = df_true.rename(columns={"Volume": "Volume_true"})
        df_pred = pd.read_csv(path_volume_pred)
    
    else:       
        path_volume_true = output_folder_area / 'volume_test.csv' #measured volume of test set
        path_volume_pred = output_folder_area / f'area_{number_of_images}_imgs.csv' #predicted volumes of whole set
        path_training = output_folder_area / f'area_volume_train_{number_of_images}_imgs.csv' #prediction of train set
        df_true = pd.read_csv(path_volume_true)
        df_true = df_true.rename(columns={"Volume": "Volume_true"})
        df_pred = pd.read_csv(path_volume_pred)
        df_training = pd.read_csv(path_training)
        df_training = df_training[df_training["Number_of_Ones"] >= 1]


    # Create a dataframe for measured and predicted test set spikes 
    if baseline_geometric: 
        df_pred['Plant_id'] = df_pred['img_id'].apply(get_plant_id) 
        # Sort by Filename to ensure consistent ordering
        df_pred = df_pred.sort_values(by=['Plant_id', 'img_id'])
        df_pred = df_pred[df_pred["volume"] > 1000]
        df_pred = df_pred[df_pred["volume"] < 35000]


        # Take first N images per plant
        df_filtered = df_pred.groupby("Plant_id").head(number_of_images)
        print(df_filtered)

        # Keep only groups that actually have N images
        group_counts = df_filtered.groupby("Plant_id")["volume"].count()
        valid_ids = group_counts[group_counts == number_of_images].index
        df_filtered = df_filtered[df_filtered["Plant_id"].isin(valid_ids)]

        
        # Now group and compute mean
        df_grouped = df_filtered.groupby("Plant_id")["volume"].mean().reset_index()

        # Count number of elements per Plant_id
        group_counts = df_filtered.groupby("Plant_id")["volume"].count().reset_index(name="count")

        print(group_counts)
        print(group_counts["count"].describe())

        
        #create a dataframe with measured and predicted volumes of test set since df_grouped contains predictions of all spikes 
        merged_df = pd.merge(df_true, df_grouped, on='Plant_id', how='inner')
        merged_df = merged_df[merged_df["volume"] >= 1]
        volume_true=np.array(merged_df['Volume_true'])
        volume_pred=np.array(merged_df['volume'])

    else:
        #create a dataframe with measured and predicted volumes of test set since df_pred contains predictions of all spikes 
        merged_df = pd.merge(df_true, df_pred, on='Plant_id', how='inner')
        merged_df = merged_df[merged_df["Number_of_Ones"] >= 1]

        volume_true=np.array(merged_df['Volume_true'])
        volume_pred=np.array(merged_df['Number_of_Ones'])



    #Get the measured and predicted volumes of the training set to calculate a scaling factor
    if baseline_geometric: 

        #get training predictions:
        volume_train_pred = pd.merge(df_train_true, df_grouped, on='Plant_id', how='inner')
        volume_train_pred = volume_train_pred[volume_train_pred["volume"] >= 1]

        train_true_np = np.array(volume_train_pred['Volume_train'])
        train_pred_np = np.array(volume_train_pred['volume'])

        # ----------------------------------------------------
        # 2) Fit models: TRUE ≈ f(PRED)
        # ----------------------------------------------------
        X = train_pred_np.reshape(-1, 1)   # input = predicted
        y = train_true_np                  # target = true

        # --- Linear model ---
        lin_model = LinearRegression()
        lin_model.fit(X, y)
        y_hat_lin = lin_model.predict(X)

        # --- Quadratic model ---
        poly2_model = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression()
        )
        poly2_model.fit(X, y)
        y_hat_poly2 = poly2_model.predict(X)

        # --- Cubic model ---
        poly3_model = make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False),
            LinearRegression()
        )
        poly3_model.fit(X, y)
        y_hat_poly3 = poly3_model.predict(X)

        # --- Exponential model: true ≈ c * exp(b * pred) ---
        mask = (train_pred_np > 0) & (train_true_np > 0)
        X_exp = X[mask]
        y_exp = y[mask]

        logy = np.log(y_exp)
        exp_model = LinearRegression()
        exp_model.fit(X_exp, logy)
        logy_hat = exp_model.predict(X_exp)
        y_hat_exp = np.exp(logy_hat)

        # ----------------------------------------------------
        # 3) Compute R² for all
        # ----------------------------------------------------
        r2_lin   = r2_score(y, y_hat_lin)
        r2_poly2 = r2_score(y, y_hat_poly2)
        r2_poly3 = r2_score(y, y_hat_poly3)
        r2_exp   = r2_score(y_exp, y_hat_exp)

        # ----------------------------------------------------
        # 4) Plot: TRUE (y) vs PREDICTED (x)
        # ----------------------------------------------------
        x_min, x_max = 0, 16_000   # predicted volume range
        y_min, y_max = 0, 11_000   # true volume range

        out_file = output_folder_geometric / f"train_pred_vs_true_{number_of_images}_fits.png"
        plt.figure(figsize=(6, 6))
        plt.scatter(train_pred_np, train_true_np, alpha=0.5, label="training data")

        # 1:1 line (perfect calibration)
        line_max = min(x_max, y_max)
        plt.plot([x_min, line_max], [x_min, line_max], linestyle="--", label="y = x")

        # smooth curves
        x_curve = np.linspace(x_min, x_max, 200).reshape(-1, 1)
        y_curve_lin   = lin_model.predict(x_curve)
        y_curve_poly2 = poly2_model.predict(x_curve)
        #y_curve_poly3 = poly3_model.predict(x_curve)
        y_curve_exp   = np.exp(exp_model.predict(x_curve))

        # plot all fits
        plt.plot(x_curve, y_curve_lin,   label=f"linear (R²={r2_lin:.3f})")
        plt.plot(x_curve, y_curve_poly2, label=f"quadratic (R²={r2_poly2:.3f})")
        #plt.plot(x_curve, y_curve_poly3, label=f"cubic (R²={r2_poly3:.3f})")
        plt.plot(x_curve, y_curve_exp,   label=f"exponential (R²={r2_exp:.3f})")

        plt.xlabel("Predicted Volume")
        plt.ylabel("True Volume")
        plt.title("Calibration: true vs predicted volume\nlinear, quadratic, cubic, exponential")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close()

    
        # ----------------------------------------------------
        # 5) Apply cubic calibration to TEST predictions
        # ----------------------------------------------------
        y_pred_rescaled = poly2_model.predict(volume_pred.reshape(-1, 1))

        # --- quadratic ---
        lin2 = poly2_model.named_steps['linearregression']
        coefs2 = lin2.coef_.ravel()
        intercept2 = lin2.intercept_

        print("\nQuadratic calibration function:")
        print(f"y_true = {intercept2:.6f} + "
            f"{coefs2[0]:.6f}·x + "
            f"{coefs2[1]:.6f}·x²")


        path_merged_file = output_folder_geometric / f"merged_{number_of_images}_imgs.csv"
        merged_df["volume_predicted_scaled"] = y_pred_rescaled
        merged_df.to_csv(path_merged_file, index=False)

        # -----------------------------------------
        # PLOT: TRAIN CURVE + TEST SCATTER OVERLAID
        # -----------------------------------------
        plt.figure(figsize=(7, 7))

        # --- training scatter ---
        plt.scatter(train_pred_np, train_true_np, s=15, alpha=1, label="train data", color="gray")

        # --- 1:1 line ---
        line_max = min(x_max, y_max)
        plt.plot([x_min, line_max], [x_min, line_max], "k--", label="y = x")

        # --- calibration curves ---
        x_curve = np.linspace(x_min, x_max, 300).reshape(-1, 1)
        plt.plot(x_curve, lin_model.predict(x_curve),label=f"linear (train) (R²={r2_lin:.3f})",color="darkblue",)
        plt.plot(x_curve, poly2_model.predict(x_curve),label=f"quadratic (train) (R²={r2_poly2:.3f})",color="red",)
        plt.plot(x_curve, np.exp(exp_model.predict(x_curve)),label=f"exponential (train) (R²={r2_exp:.3f})",color="darkred",)

        # --- test scatter ---
        plt.scatter(volume_pred, volume_true, s=15,alpha=1, label="test data", color="skyblue")

        # axis + labels
        plt.xlabel("Predicted Volume", fontsize = 15)
        plt.ylabel("True Volume", fontsize = 15)
        #plt.title("Calibration curves (train) + test set overlay")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0, 5, 1, 2, 3, 4]
        plt.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", bbox_to_anchor=(0.02, 0.98),  frameon=True,)

        plt.grid(True)

        # save
        plt.tight_layout()
        plt.savefig(output_folder_geometric / f"train_curves_plus_test_overlay_{number_of_images}.png", dpi=300)


    else:
        train_true_np = np.array(df_training['Volume'])
        train_pred_np = np.array(df_training['Number_of_Ones'])

        # ----------------------------------------------------
        # 2) Fit models: TRUE ≈ f(PRED)
        # ----------------------------------------------------
        X = train_pred_np.reshape(-1, 1)   # input = predicted
        y = train_true_np                  # target = true

        # --- Linear model ---
        lin_model = LinearRegression()
        lin_model.fit(X, y)
        y_hat_lin = lin_model.predict(X)

        # --- Quadratic model ---
        poly2_model = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression()
        )
        poly2_model.fit(X, y)
        y_hat_poly2 = poly2_model.predict(X)

        # --- Cubic model ---
        poly3_model = make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False),
            LinearRegression()
        )
        poly3_model.fit(X, y)
        y_hat_poly3 = poly3_model.predict(X)

        # --- Exponential model: true ≈ c * exp(b * pred) ---
        mask = (train_pred_np > 0) & (train_true_np > 0)
        X_exp = X[mask]
        y_exp = y[mask]

        logy = np.log(y_exp)
        exp_model = LinearRegression()
        exp_model.fit(X_exp, logy)
        logy_hat = exp_model.predict(X_exp)
        y_hat_exp = np.exp(logy_hat)

        # ----------------------------------------------------
        # 3) Compute R² for all
        # ----------------------------------------------------
        r2_lin   = r2_score(y, y_hat_lin)
        r2_poly2 = r2_score(y, y_hat_poly2)
        r2_poly3 = r2_score(y, y_hat_poly3)
        r2_exp   = r2_score(y_exp, y_hat_exp)

        # ----------------------------------------------------
        # 4) Plot: TRUE (y) vs PREDICTED (x)
        # ----------------------------------------------------
        x_min, x_max = 0, 750_000   # predicted volume range
        y_min, y_max = 0, 10_000   # true volume range

        out_file = output_folder_area / f"train_pred_vs_true_{number_of_images}_fits.png"
        plt.figure(figsize=(6, 6))
        plt.scatter(train_pred_np, train_true_np, alpha=0.5, label="training data")

        # 1:1 line (perfect calibration)
        #line_max = min(x_max, y_max)
        #plt.plot([x_min, line_max], [x_min, line_max], linestyle="--", label="y = x")

        # smooth curves
        x_curve = np.linspace(x_min, x_max, 200).reshape(-1, 1)
        y_curve_lin   = lin_model.predict(x_curve)
        y_curve_poly2 = poly2_model.predict(x_curve)
        #y_curve_poly3 = poly3_model.predict(x_curve)
        y_curve_exp   = np.exp(exp_model.predict(x_curve))

        # plot all fits
        plt.plot(x_curve, y_curve_lin,   label=f"linear (R²={r2_lin:.3f})")
        plt.plot(x_curve, y_curve_poly2, label=f"quadratic (R²={r2_poly2:.3f})")
        #plt.plot(x_curve, y_curve_poly3, label=f"cubic (R²={r2_poly3:.3f})")
        plt.plot(x_curve, y_curve_exp,   label=f"exponential (R²={r2_exp:.3f})")

        plt.xlabel("Mean Number of Pixels")
        plt.ylabel("True Volume")
        plt.title("Calibration: true vs predicted volume\nlinear, quadratic, cubic, exponential")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close()

        # ----------------------------------------------------
        # 5) Apply cubic calibration to TEST predictions
        # ----------------------------------------------------
        y_pred_rescaled = poly2_model.predict(volume_pred.reshape(-1, 1))

        # --- quadratic ---
        lin2 = poly2_model.named_steps['linearregression']
        coefs2 = lin2.coef_.ravel()
        intercept2 = lin2.intercept_

        print("\nQuadratic calibration function:")
        print(f"y_true = {intercept2:.10f} + "
            f"{coefs2[0]:.10f}·x + "
            f"{coefs2[1]:.10f}·x²")


        path_merged_file = output_folder_area / f"merged_{number_of_images}_imgs.csv"
        merged_df["volume_predicted_scaled"] = y_pred_rescaled
        merged_df.to_csv(path_merged_file, index=False)

        # -----------------------------------------
        # PLOT: TRAIN CURVE + TEST SCATTER OVERLAID
        # -----------------------------------------
        plt.figure(figsize=(7, 7))

        # --- training scatter ---
        plt.scatter(train_pred_np, train_true_np, s=15, alpha=1, label="train data", color="gray")

        # --- 1:1 line ---
        #line_max = min(x_max, y_max)
        #plt.plot([x_min, line_max], [x_min, line_max], "k--", label="y = x")

        # --- calibration curves ---
        x_curve = np.linspace(x_min, x_max, 300).reshape(-1, 1)
        plt.plot(x_curve, lin_model.predict(x_curve),label=f"linear (train) (R²={r2_lin:.3f})",color="darkblue",)
        plt.plot(x_curve, poly2_model.predict(x_curve),label=f"quadratic (train) (R²={r2_poly2:.3f})",color="red",)
        plt.plot(x_curve, np.exp(exp_model.predict(x_curve)),label=f"exponential (train) (R²={r2_exp:.3f})",color="darkred",)

        # --- test scatter ---
        plt.scatter(volume_pred, volume_true, s=15, alpha=1, label="test data", color="skyblue")

        # axis + labels
        plt.xlabel("Mean Number of Pixels", fontsize = 15)
        plt.ylabel("True Volume", fontsize = 15)
        #plt.title("Calibration curves (train) + test set overlay")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0, 4, 1 , 2, 3]
        plt.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", bbox_to_anchor=(0.02, 0.98),  frameon=True,)
        plt.grid(True)

        # save
        plt.tight_layout()
        plt.savefig(output_folder_area / f"train_curves_plus_test_overlay_{number_of_images}.png", dpi=300)




    # Calculate the correlation, r2 and mape between the measured volume and the scaled predictions
    r2_score_rescaled = r2_score(volume_true, y_pred_rescaled)
    print("r2 score: ")
    print(r2_score_rescaled)

    correlation = np.corrcoef(volume_true, y_pred_rescaled)
    cor = correlation[0,1]
    print("correlation: ")
    print(cor)

    MAPE = mean_absolute_percentage_error(volume_true, y_pred_rescaled)
    print("mape: ")
    print(MAPE)

    MAE = mean_absolute_error(volume_true, y_pred_rescaled)
    print("mae:")
    print(MAE)

    #################################################
    #Plot the correlation

    first_digit_2023 = [21,23]
    second_digit_2023 = [6,7,8,9,10]
    second_digit_2024 = [15, 16, 17, 18, 19]


    def get_sampling_date(plant_id):
        first_digit = int(plant_id.split('_')[0])
        second_digit = int(plant_id.split('_')[1])
        if second_digit in second_digit_2023 or first_digit in first_digit_2023:
            last_digit = int(plant_id.split('_')[-1])  # Extract the last digit
            if last_digit in [1, 2, 10]:
                return 1
            elif last_digit in [3, 4]:
                return 2
            elif last_digit in [5, 6, 7, 8, 9]:
                return 3
            else:
                return 0  # Handle unexpected cases
        elif second_digit in second_digit_2024: 
            last_digit = int(plant_id.split('_')[-1])  # Extract the last digit
            
            if last_digit in [1, 2, 3]:
                return 4
            elif last_digit in [4, 5]:
                return 5
            elif last_digit in [6, 7, 8, 9, 10]:
                return 6
            else:
                return 0  # Handle unexpected cases

    merged_df["sampling_date"] = merged_df["Plant_id"].apply(get_sampling_date)
    # Define colors based on sampling_date
    color_map = {0: 'black', 
                 1: 'darkblue', 
                 2: 'Purple', 
                 3: 'darkgreen', 
                 4: 'lightblue', 
                 5: 'orchid', 
                 6: 'lightgreen'}  

    # Assign colors based on sampling_date
    colors = merged_df['sampling_date'].map(color_map)
    alpha_values = 1
    plt.figure(figsize=(8, 6))

    if baseline_geometric: 
        plt.scatter(merged_df['Volume_true'], y_pred_rescaled, color=colors, alpha = alpha_values, label="Sampling date")
        plt.plot(merged_df['Volume_true'], merged_df['Volume_true'], 'k--', label='x = y')
    else: 
        plt.scatter(merged_df['Volume_true'], y_pred_rescaled, color=colors, alpha = alpha_values, label="Sampling date")
        plt.plot(merged_df['Volume_true'], merged_df['Volume_true'], 'k--', label='x = y')

    # Labeling
    plt.xlabel('True Volume', fontsize = 18)
    plt.ylabel('Predicted Volume', fontsize=18)
    plt.title('Predicted Volume vs True Volume', fontsize = 18)
    # Add the correlation coefficient and R² score to the plot
    #plt.text(0.065, 0.95, f'r: {cor:.2f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    #plt.text(0.065, 0.90, f'R²: {r2_score_rescaled:.2f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')

    props = {'boxstyle': 'round', 'facecolor': 'white', 'edgecolor': 'black', 'alpha': 1}
    textstr = f'r: {cor:.2f}\nR²: {r2_score_rescaled:.2f}\nMAPE: {round(MAPE,2)}\nMAE: {MAE:.2f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    # Create a manual legend for categorical colors
    legend_patches = [
        mpatches.Patch(color='darkblue', label='1st Sampling Date 2023'),
        mpatches.Patch(color='purple', label='2nd Sampling Date 2023'),
        mpatches.Patch(color='darkgreen', label='3rd Sampling Date 2023'), 
        mpatches.Patch(color='lightblue', label='1st Sampling Date 2024'),
        mpatches.Patch(color='orchid', label='2nd Sampling Date 2024'),
        mpatches.Patch(color='lightgreen', label='3rd Sampling Date 2024')
    ]
    plt.legend(handles=legend_patches, title="Sampling Date", loc="lower right")

    # Show plot
    plt.grid(True)
    plt.show()

    if baseline_geometric:
        path_plot_save =  output_folder_geometric / f'plot_geometric_baseline_{number_of_images}_imgs.png'
        plt.savefig(path_plot_save)
    else: 
        path_plot_save = output_folder_area / f'plot_area_baseline_{number_of_images}_imgs.png'
        plt.savefig(path_plot_save)




@click.command()
@click.option("--output_area", type=str, required=True, help="Path to output folder of the script area_baseline.py. Must exist!")
@click.option("--input_geom", type=str, required=True, help="Path to output folder of the geometric baseline with prediction csv. Must exist!")
@click.option("--output_geom", type=str, required=True, help="Path to output folder of the geometric baseline with prediction csv. Will be created if it does not exist.")
@click.option("--mapping_train", type=str, required=True, help="Path to a train mapping file with a desired split.")
@click.option("--num_imgs", type=int, required=True, help="Number of images per plant that should be considered.")
@click.option("--geom", type=bool, required=True, help="Calculates metrics for geometric baseline if True, else for area baseline")


def main(output_area: str, 
         input_geom: str,
         output_geom: str, 
         mapping_train: str, 
         num_imgs: int, 
         geom:bool):
    
    output_area = Path(output_area)
    input_geom = Path(input_geom)
    output_geom = Path(output_geom)
    output_geom.mkdir(parents=True, exist_ok=True)

    metrics_baseline(
                     output_folder_area=output_area, 
                     input_folder_geometric=input_geom,
                     output_folder_geometric=output_geom,
                     mapping_train=mapping_train, 
                     number_of_images=num_imgs,
                     baseline_geometric=geom)

if __name__ == "__main__":
    main()





