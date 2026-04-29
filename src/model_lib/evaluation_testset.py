import data_lib.data_sets as data_sets
import lib.stats_lib as stats_lib
import pandas as pd
import torch
from torchvision.transforms import v2
from icecream import ic
from model_lib.inferrer import Inferrer, ExperimentCreator
from pathlib import Path
from model_lib.inferrer import MultiImageExperimentCreator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def evaluation(dataset_path : Path | str,
              pretrained_transform : list,
              output_path : Path | str,
              mapping_file : Path | str,
              value_column : str,
              Y_STD : float, 
              Y_MEAN : float,
              pretrained_model : torch.nn.Module,
              max_sequence_length : int,
              model, 
              experiment_creator : ExperimentCreator,
              verbose : bool = False):
    """Compute one training run including training, inference and computation of stats for a dataset that only contains testdata!

    Args:
        dataset_path (Path | str): Directory of the dataset
        pretrained_transform (list): Model specific transform settings which should include crop, normalize and resize.
        output_path (Path | str): Directory to store all the output
        mapping_file (Path | str): Csv file containing Image -> Plant / value mapping 
        value_column (str): Column in mapping file to be regarded as value
        Y_MEAN (float): Mean volume of training dataset 
        Y_STD (float): Standard deviation of volume of training dataset 
        pretrained_model (torch.nn.Module): Pretrained model to extract features from
        max_sequence_length (int): Maximal seqence lenght for one plant
        random_seqence_length (bool): Whether or not to use random seqence lenth
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
    save_mapping_test = output_path / "mapping_test.json"
    save_predictions_path = output_path / "predictions.json"

    # create / get experiment
    plant_mapping = pd.read_json(save_mapping_test, orient='index', convert_axes=False)
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
   
    # calculate metrics:
    # Make sure plant_ids are not converted to int in a weird manner
    df_pred = pd.read_json(save_predictions_path, orient='index' , dtype={'plant_id' : str})

    # The index of the mapping files are plant_ids so convert_axes = False otherwise it's converted to int
    df_test = pd.read_json(save_mapping_test, orient='index', convert_axes=False)

    # prints and saves statistics
    charts_output_path = output_path    
    fig_test, stat_test = stats_lib.compare(mapping_true=df_test,
                                                mapping_predicted=df_pred,
                                                output_name=value_column,
                                                name="test",
                                                output_folder=charts_output_path)
    
    
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
    
    #merge the true and predicted dataframe to keep only the ones in the test set
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

    #Plot 
    alpha_values = 1
    plt.figure(figsize=(8, 6))
    plt.scatter(merged_df['volume_true'], merged_df['volume_predicted'], alpha = alpha_values, label="Sampling date")
    plt.plot(merged_df['volume_true'], merged_df['volume_true'], 'r--', label='x = y')
    plt.xlabel('True Volume', fontsize = 18)
    plt.ylabel('Predicted Volume', fontsize=18)
    plt.title('Predicted Volume vs True Volume', fontsize = 18)
    props = {'boxstyle': 'round', 'facecolor': 'white', 'edgecolor': 'black', 'alpha': 1}
    textstr = f'r: {cor:.2f}\nR²: {r2_score_rescaled:.2f}\nMAPE: {round(MAPE,2)}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    plt.grid(True)
    plt.show()
    plt.savefig(f'data/field_images_trained_labimages/mlp_run_0/results/plot_best_{model_name}_colors_field_trained_on_good_images.png')


model_transform = [v2.Resize(size=256, antialias=True),
                       v2.CenterCrop(224),
                       v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
#input parameters 
eval_max_sequence_len = 1
eval_min_sequence_len = 1
eval_random_sequence_len = False
eval_random_choice = False

pretrained_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")     
experiment_creator = MultiImageExperimentCreator(ignore_artificial=True, 
                                                             eval_min_seq_len=eval_min_sequence_len, 
                                                             eval_max_seq_len=eval_max_sequence_len,
                                                             eval_random_seq_len=eval_random_sequence_len, 
                                                             random_choice=eval_random_choice)


evaluation(dataset_path="/projects/zumstego/1_datasets/field_images_no_bar_crop",
                        pretrained_transform=model_transform, 
                        output_path="data/field_images_trained_labimages/mlp_run_0/results", #adapt
                        mapping_file="/projects/zumstego/1_datasets/field_images_no_bar_crop/vol_mapping.csv",
                        value_column="volume",
                        Y_STD = 1235.42, 
                        Y_MEAN = 4769.53,
                        max_sequence_length=1,  #ADAPT
                        model = "data/check_1imgs_scaled_try2/mlp_run_0/model_21.pth", #adapt
                        pretrained_model=pretrained_model,
                        experiment_creator=experiment_creator,
                        verbose=False)




























