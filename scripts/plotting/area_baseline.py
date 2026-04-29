import numpy as np 
import pandas as pd
import json
import click
import matplotlib.pyplot as plt
from pathlib import Path


def area_baseline(input_folder: str, 
                  output_folder_area: str, 
                  mapping_test: str, 
                  mapping_train: str, 
                  number_of_images: int): 

    """
    Goes through the npz files of the input 
    folder and sums up the number of foreground 
    pixels (ones). Takes the average of the
    pixels per plant and merges the dataframe to 
    the test plants from the test mapping, keeping only 
    test plants. 
    Also filters the plants from the training data set in order 
    to calculate a shifting factor later on. 
    
    Args: 
        input_folder (str): Path to input folder 
        output_folder (str): Path to output folder
        mapping_test (str): Path to test mapping file
        mapping_train (str): Path to train mapping file
        number_of_images (int): Number of images per spike considered in calculation of mean
        number of pixles per spike
    """

    #create output folder in case it does not exist
    output_folder_area.mkdir(parents=True, exist_ok=True)

    ################# test set ##########################

    # Load the test mapping file to get the spikes that we want to consider
    with open(mapping_test, 'r') as f:
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
    df_json = pd.DataFrame({'Plant_id': plant_ids, 'Volume': volumes})

    output_file = output_folder_area / "volume_test.csv"

    #save the dataframe
    df_json.to_csv(output_file, index=False)


    ################# training set ##########################
    # Get the training set and filter the spikes of the training in order 
    # to filter the spikes in the training file. Will be used in file
    # metrics_baseline.py to calculate a scaling factor 

    # Load the JSON file
    with open(mapping_train, 'r') as f:
        data_train_true = json.load(f)
        
    # Initialize empty lists to store the data
    plant_ids_train_true = []
    volumes_train_true = []

    # Iterate through each entry in the JSON data
    for plant_id, info in data_train_true.items():
        # Extract the 'volume' information
        volume_train_true = info['volume']
        
        # Append the 'Plant_id' and 'volume' to the lists
        plant_ids_train_true.append(plant_id)
        volumes_train_true.append(volume_train_true)

    # Create a dataframe from the lists
    df_json_train_true = pd.DataFrame({'Plant_id': plant_ids_train_true, 'Volume': volumes_train_true})

    #########################################################################

    #go through npz files 
    directory = input_folder

    # Initialize an empty list to store the data
    data_list = []

    # Iterate through each .npz file in the directory
    for file_path in directory.glob("*.npz"):
        
        filename = file_path.name

        try: 
            data = np.load(file_path)

        except Exception as e: 
            print(f'skipping {filename}: {e}')
            continue
        
        array = data['arr_0']

        # Convert boolean values to integers (True -> 1, False -> 0)
        array_int = array.astype(int)

        # Compute the sum of ones in the matrix
        sum_of_ones = np.sum(array_int)

        # Append the filename and the sum of ones to the data list
        data_list.append({'Filename': filename, 'Number_of_Ones': sum_of_ones})
        
        # Close the .npz file
        data.close()

    

    # Create a dataframe from the data list
    df = pd.DataFrame(data_list)

    # Remove the '.npz' extension from the filenames
    df['Filename'] = df['Filename'].str.rstrip('.npz')

    # Create a new column with the modified filenames
    df['Plant_id'] = df['Filename'].str.rsplit('_', n=2).str[0]

    # Sort by Filename to ensure consistent ordering
    df = df.sort_values(by=['Plant_id', 'Filename'])

     #remove files where segmentation failed
    df = df[df["Number_of_Ones"] > 100000]
    df = df[df["Number_of_Ones"] < 650000]


    # For each plant_id, keep only the first entries (sorted by filename)
    df_filtered = df.groupby('Plant_id').head(number_of_images)

    # Keep only groups that actually have N images
    print(df_filtered)
    group_counts = df_filtered.groupby("Plant_id")["Number_of_Ones"].count()
    valid_ids = group_counts[group_counts == number_of_images].index
    df_filtered = df_filtered[df_filtered["Plant_id"].isin(valid_ids)]


    # Group the rows based on the modified filename and calculate the mean of 'Number_of_Ones'
    df_grouped = df_filtered.groupby('Plant_id')['Number_of_Ones'].mean().reset_index()
    output_file_area = output_folder_area / f"area_{number_of_images}_imgs.csv"
    df_grouped.to_csv(output_file_area, index=False)

    ###################################################################
    #keep spikes in test and training 

    # Merge the dataframes based on the 'Plant_id' column --> only keep plants in test set
    merged_df = pd.merge(df_json, df_grouped, on='Plant_id', how='inner')

    #save the dataframe
    path_merged_df = output_folder_area / f"area_volume_test_{number_of_images}_imgs.csv"
    merged_df.to_csv(path_merged_df, index=False)

    #do the same for training data set: keep predictions
    # Merge the dataframes based on the 'Plant_id' column --> only keep plants in training set
    merged_df = pd.merge(df_json_train_true, df_grouped, on='Plant_id', how='inner')

    #save the dataframe
    path_merged_df_train = output_folder_area / f"area_volume_train_{number_of_images}_imgs.csv"
    merged_df.to_csv(path_merged_df_train, index=False)

    ################################################

    #plot distribution of test spikes 
    plt.hist(merged_df["Number_of_Ones"], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of {}'.format("Number_of_Ones"))
    plt.grid(True)
    plt.show()

    plot_distribution = output_folder_area / f'distribution_plot_{number_of_images}_imgs.png'
    plt.savefig(plot_distribution)



@click.command()
@click.option("--input", type=str, required=True, help="Path to image dataset folder containing npz files.")
@click.option("--output", type=str, required=True, help="Path to output folder. Will be created if it does not exist")
@click.option("--mapping_test", type=str, required=True, help="Path to a test mapping file")
@click.option("--mapping_train", type=str, required=True, help="Path to a train mapping file.")
@click.option("--num_imgs", type=int, required=True, help="Number of images per plant that should be considered.")

def main(input: str, output: str, mapping_test: str, mapping_train: str, num_imgs: int):

    input = Path(input)
    output = Path(output)

    area_baseline(input_folder=input, output_folder_area=output, mapping_test=mapping_test, mapping_train=mapping_train, number_of_images=num_imgs)

if __name__ == "__main__":
    main()



