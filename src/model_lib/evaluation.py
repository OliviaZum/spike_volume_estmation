import data_lib.data_sets as data_sets
import lib.stats_lib as stats_lib
import pandas as pd
import torch
from torchvision.transforms import v2
from icecream import ic
from model_lib.inferrer import Inferrer, ExperimentCreator
from pathlib import Path
from model_lib.inferrer import  MultiImageExperimentCreator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.patches as mpatches
import time 
import torchvision.models as models
import torch.nn as nn
import timm
import shutil

def evaluation(dataset_path : Path | str,
              pretrained_transform : list,
              output_path : Path | str,
              mapping_file : Path | str,
              value_column : str,
              Y_STD : float, 
              Y_MEAN : float,
              max_sequence_length : int,
              pretrained_model : torch.nn.Module,
              model, 
              experiment_creator : ExperimentCreator,
              verbose : bool = False):
    """Compute one training run including training, inference and computation of stats

        dataset_path (Path | str): Directory of the dataset
        pretrained_transform (list): Model specific transform settings which should include crop, normalize and resize.
        output_path (Path | str): Directory to store all the output
        mapping_file (Path | str): Csv file containing Image -> Plant / value mapping 
        value_column (str): Column in mapping file to be regarded as value
        Y_MEAN (float): Mean volume of training dataset 
        Y_STD (float): Standard deviation of volume of training dataset 
        pretrained_model (torch.nn.Module): Pretrained model to extract features from
        max_sequence_length (int): Maximal seqence lenght for one plant
        random_seqence_lenght (bool): Whether or not to use random seqence lenth
        model (torch.nn.Module): Model to be trained
        experiment_creator (ExperimentCreator): Experiment creator for inference
        verbose (bool, optional): Defaults to False
    """

    #extract model name for image saving
    model_name = model.split("/")[-1].split(".")[0]
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    mapping_file = Path(mapping_file)

    # compute output paths for mapping files and predictions
    save_mapping_plants = output_path / "mapping_all_plants.json"
    save_mapping_train = output_path / "mapping_train.json"
    save_mapping_val = output_path / "mapping_val.json"
    save_mapping_test = output_path / "mapping_test.json"
    save_predictions_path = output_path / "predictions.json"

    #copy json files into results folder
    for filename in ["mapping_all_plants.json","mapping_train.json","mapping_val.json","mapping_test.json","predictions.json",]:
        src = output_path.parent / filename
        dst = output_path / filename

        if src.exists():
            shutil.move(str(src), str(dst))

    # create / get experiment
    plant_mapping = pd.read_json(save_mapping_plants, orient='index', convert_axes=False)
    experiment = experiment_creator.create_experiment(plant_mapping=plant_mapping,)
    # Compute Predictions for all data including test data

    
    inferrer = Inferrer(saved_model_path=model, 
                            output_name=value_column,
                            Y_STD=Y_STD, 
                            Y_MEAN=Y_MEAN,
                            pretrained_model=pretrained_model,
                            verbose=verbose)
    
    dataset = data_sets.ImageInferenceDataset(image_dir=dataset_path,
                                              transform=pretrained_transform,
                                              model_sequence_size=max_sequence_length,
                                              experiment=experiment)
    
    #comptes predictions for the whole dataset (training, test, evaluation)
    experiment_output = inferrer.infer_experiment(dataset=dataset)
    experiment_output.to_json(save_predictions_path, orient='index')

    # Calculate metrics:
    # Make sure plant_ids are not converted to int in a weird manner
    df_pred = pd.read_json(save_predictions_path, orient='index' , dtype={'plant_id' : str})

    # The index of the mapping files are plant_ids so convert_axes = False, otherwise it's converted to int
    df_train = pd.read_json(save_mapping_train, orient='index', convert_axes=False)
    df_test = pd.read_json(save_mapping_test, orient='index', convert_axes=False)
    df_val = pd.read_json(save_mapping_val, orient='index', convert_axes=False)

    # prints and saves statistics
    charts_output_path = output_path 
    
    fig_train, stat_train = stats_lib.compare(mapping_true=df_train,
                                            mapping_predicted=df_pred,
                                            output_name=value_column,
                                            name="train",
                                            output_folder=charts_output_path,
                                            verbose=verbose)
    fig_test, stat_test = stats_lib.compare(mapping_true=df_test,
                                                mapping_predicted=df_pred,
                                                output_name=value_column,
                                                name="test",
                                                output_folder=charts_output_path)

    fig_val, stat_val = stats_lib.compare(mapping_true=df_val,
                                        mapping_predicted=df_pred,
                                        output_name=value_column,
                                        name="val",
                                        output_folder=charts_output_path, 
                                        verbose=verbose)

    

    ##############################################
    # Plot testset
    ##############################################
    
    # Rename the first  column of dataframe with true values to 'plant_id'
    df_test.index.name = 'plant_id'
    df_test.reset_index(inplace=True)
    
    #rename column names of predicted dataframe to avoid confusion, add "predicted"
    df_test = df_test.rename(columns=lambda x: x + '_true' if x != 'plant_id' else x)
    
    #rename column names of predicted dataframe to avoid confusion, add "predicted"
    df_pred = df_pred.rename(columns=lambda x: x + '_predicted' if x != 'plant_id' else x)
    
    #merge the true and predicted dataframe
    merged_df = pd.merge(df_test, df_pred, on='plant_id', how='left')
    
    # Calculate the correlation, r2 and mape between the 'Volume' and 'predicted volume' columns
    volume_true=np.array(merged_df['volume_true'])
    volume_predicted=np.array(merged_df['volume_predicted'])
   
    r2_score_rescaled = r2_score(volume_true, volume_predicted)
    print(f'r2: {r2_score_rescaled}')

    correlation = np.corrcoef(volume_true, volume_predicted)
    cor = correlation[0,1]
    print(f'cor: {cor}')
    
    def mean_absolute_percentage_error(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            mask = y_true != 0  # Avoid division by zero
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        

    MAPE = mean_absolute_percentage_error(volume_true, volume_predicted)
    print(f'mape: {MAPE}')

    MAE = np.mean(np.abs(volume_true - volume_predicted))
    print(f'MAE: {MAE}')


    # Plot
    first_digit_2023 = [21,23]
    second_digit_2023 = [6,7,8,9,10]
    second_digit_2024 = [15, 16, 17, 18, 19]
    
    def get_sampling_date(plant_id):
        first_digit = int(plant_id.split('_')[0])
        second_digit = int(plant_id.split('_')[1])
        last_digit = int(plant_id.split('_')[-1])  # Extract the last digit

        if first_digit in first_digit_2023: 
            if last_digit in [5,6]:
                return '2023-06-10'
            elif last_digit in [3, 4]:
                return '2023-07-04'
            elif last_digit in [1,2]:
                return '2023-07-11a'

        if second_digit in second_digit_2023 :
            
            if last_digit in [1, 2, 10]:
                return '2023-06-09'
            elif last_digit in [3, 4]:
                return '2023-06-29'
            elif last_digit in [5, 6, 7, 8, 9]:
                return '2023-07-11b'
            else:
                return 0  # Handle unexpected cases
        elif second_digit in second_digit_2024: 
            last_digit = int(plant_id.split('_')[-1])  # Extract the last digit
            
            if last_digit in [1, 2, 3]:
                return '2024-06-12'
            elif last_digit in [4, 5]:
                return '2024-07-05'
            elif last_digit in [6, 7, 8, 9, 10]:
                return '2024-07-19'
            else:
                return 0  # Handle unexpected cases

        
    merged_df["sampling_date"] = merged_df["plant_id"].apply(get_sampling_date)
    color_map = {'2023-06-10': 'darkred', '2023-07-04': 'darkorange', '2023-07-11a': 'purple',
                  '2023-06-09': 'orchid', '2023-06-29': 'lightblue', '2023-07-11b': 'darkgreen',
                    '2024-06-12': 'lightgreen', '2024-07-05' : 'darkblue', '2024-07-19': 'black'}  # Define colors for each category
    colors = merged_df['sampling_date'].map(color_map)
   
    alpha_values = 1

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(merged_df['volume_true'], merged_df['volume_predicted'], color=colors, alpha = alpha_values, label="Sampling date")
    min_val = min(merged_df['volume_true'].min(), merged_df['volume_predicted'].min())
    max_val = max(merged_df['volume_true'].max(), merged_df['volume_predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='x = y')    
    ax.set_xlabel('Measured Volume [mm³]', fontsize=17)
    ax.set_ylabel('Predicted Volume [mm³]', fontsize=17)
    ax.set_ylim(1000, None)
    #plt.xlim(1500, None)
    ax.tick_params(axis='both', labelsize=14)
    props = {'boxstyle': 'round', 'facecolor': 'white', 'edgecolor': 'black', 'alpha': 1}
    textstr = f'r: {cor:.2f}\nR²: {r2_score_rescaled:.2f}\nMAPE: {round(MAPE,2)}\nMAE: {round(MAE,2)}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    # Create a manual legend for categorical colors
    legend_patches = [
        mpatches.Patch(color='darkred', label='2023-06-10'),
        mpatches.Patch(color='darkorange', label='2023-07-04'),
        mpatches.Patch(color='purple', label='2023-07-11a'), 
        mpatches.Patch(color='orchid', label='2023-06-09'),
        mpatches.Patch(color='lightblue', label='2023-06-29'),
        mpatches.Patch(color='darkgreen', label='2023-07-11b'), 
        mpatches.Patch(color='lightgreen', label='2024-06-12'),
        mpatches.Patch(color='darkblue', label='2024-07-05'),
        mpatches.Patch(color='black', label='2024-07-19')
    ]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    legend = ax.legend(
    handles=legend_patches,
    title="Sampling Date",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=15,
    title_fontsize=13,
    frameon=True
)

    legend.get_frame().set_edgecolor('black')  
    
    ax.grid(True)
    
    plt.savefig(f'data/test_dexample_folderelete/ft_run_0/results/plot_{model_name}_example.png', dpi=1800)
    plt.show()

   
model_transform = [v2.Resize(size=256, antialias=True),
                       v2.CenterCrop(224),
                       v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

#input parameters 
eval_max_sequence_len = 1
eval_min_sequence_len = 1
eval_random_sequence_len = False
eval_random_choice = False

pretrain = "None" #resnet18 / #dinov3 /"none" / "feature"

if pretrain == "dinov2": 
    pretrained_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14") 
elif pretrain == "resnet18":
    pretrained_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    pretrained_model.fc = torch.nn.Identity()
elif pretrain == "dinov3":
    pretrained_model = timm.create_model("vit_small_patch16_dinov3",pretrained=True,num_classes=0)
elif pretrain == "None": 
    pretrained_model = None
    print("Using no pretrained model (Fine-tuned evaluation)")


experiment_creator = MultiImageExperimentCreator(ignore_artificial=True, 
                                                             eval_min_seq_len=eval_min_sequence_len, 
                                                             eval_max_seq_len=eval_max_sequence_len,
                                                             eval_random_seq_len=eval_random_sequence_len, 
                                                             random_choice=eval_random_choice)
#dataset_path="/projects/zumstego/1_datasets/images_no_bar_crop"
#dataset_path="/projects/zumstego/1_datasets/field_images_no_bar_crop"
evaluation(dataset_path="/projects/zumstego/1_datasets/images_no_bar_crop",
                        pretrained_transform=model_transform, 
                        output_path="data/example_folder/ft_run_0/results", #adapt
                        mapping_file="/projects/zumstego/1_datasets/images_no_bar_crop/vol_mapping.csv",
                        #mapping_file="/projects/zumstego/1_datasets/field_cropped/vol_mapping.csv",
                        value_column="volume",
                        Y_STD = 1235.42, 
                        Y_MEAN = 4769.53,
                        #Y_MEAN = 4714, #field images fine tuned
                        #Y_STD = 1111, #field images fine tuned
                        max_sequence_length=1,  #ADAPT
                        model = "data/example_folder/ft_run_0/model_0.pth", #adapt
                        pretrained_model=pretrained_model,
                        experiment_creator=experiment_creator,
                        verbose=False)











#1_resnet18-1imgs_mlp - 
#1_resnet18-2imgs_mlp - 
#1_resnet18-4imgs_mlp - 
#1_resnet18-6imgs_mlp - 

#1_resnet50-1imgs_mlp - 
#1_resnet50-2imgs_mlp - 
#1_resnet50-4imgs_mlp
#1_resnet50-6imgs_mlp