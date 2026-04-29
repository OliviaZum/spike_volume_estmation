import numpy as np
import pandas as pd
import warnings

from icecream import ic
from pathlib import Path
from typing import Tuple, Dict, List

def compute_plant_mapping_(df_mapping : pd.DataFrame,
                           value_column : str,) -> pd.DataFrame:
    """Create a dataframe with plant_id -> list of images,
    plant_id -> list of artificial indicator and plant_id -> value mappings.
    Using the `df_mapping` which is supposed to be a image_name -> volume and
    image_name -> plant_id mapping.
    
    Images in `df_mapping` with the 'img_name' ending on 'b' or 'c',
    are considered artificial images.

    Args:
        df_mapping (DataFrame): mapping file with image to volume mappings
        value_column (str): Name of the calude column.
    
    Raises:
        KeyError : If in `df_mapping` no column is named `value_column` or no column
            in `df_mapping` is named 'img_name'.
    
    Warnings:
        Warns if multiple different volumes are measured for one plant_id. 
        
    Examples:
        >>> df_mapping
                  img_name  volume plant_id  artificial
        0  10_10_1_2_b.jpg    5000  10_10_1        True
        1  10_10_1_3_a.jpg    5000  10_10_1       False
        2   10_9_2_2_c.jpg    4000   10_9_2        True
        3   10_9_2_3_a.jpg    4000   10_9_2       False
        4    8_9_2_2_c.jpg    3000    8_9_2        True
        5    8_9_2_3_a.jpg    3000    8_9_2       False
        >>> compute_plant_mapping_(df_mapping=df_mapping, value_column='volume')
                volume                              images artificial_mask
        10_10_1  5000.0  [10_10_1_2_b.jpg, 10_10_1_3_a.jpg]   [True, False]
        10_9_2   4000.0    [10_9_2_2_c.jpg, 10_9_2_3_a.jpg]   [True, False]
        8_9_2    3000.0      [8_9_2_2_c.jpg, 8_9_2_3_a.jpg]   [True, False]

    """
    
    # compute plant_ids from image names

    if df_mapping["img_name"].str.contains(r"[a-zA-Z](?=\.jpg)").any():
        plant_id = df_mapping['img_name'].str.extract(r'^(\d{1,2}_\d{1,2}_\d{1,2})_\d{1,3}(_\d{1,3})?_(\w)')
        df_mapping['plant_id'] = plant_id.iloc[:,0]
        artificial_grayscale = plant_id.iloc[:,2].str.match('b')
        artificial_color = plant_id.iloc[:,2].str.match('c')
        df_mapping['artificial'] = artificial_color | artificial_grayscale
    else: 
        plant_id = df_mapping['img_name'].str.extract(r'^(\d{1,2}_\d{1,2}_\d{1,2})_\d+')
        print(plant_id)
        df_mapping['plant_id'] = plant_id.iloc[:,0]
        df_mapping['artificial'] = False

    #compute genotype_ids from image names
    df_mapping['genotype_id'] = plant_id.iloc[:, 0].str.split('_').str[:2].str.join('_')

    # get unique plant_ids and assign plant_index
    grouping = df_mapping.loc[:,[value_column, "plant_id"]].groupby(by='plant_id')

    var = grouping.var()
    if var.shape[0] > 1 and not (var.loc[:,value_column] == 0).all():
        warnings.warn(f"Some plants have inconsitent values.")

    df_unique_mapping = grouping.mean()
    df_unique_mapping = df_unique_mapping.rename_axis('plant_id').reset_index() # create plant_id
    df_unique_mapping = df_unique_mapping.rename_axis('plant_index').reset_index() # create plant_index column
    
    # create dictonary with plant -> (value, [images], [artificial_mask]) mapping
    plants : Dict[str, Dict[str, any]] = {}
    for _ , current_plant in df_unique_mapping.iterrows():
        slic_ = df_mapping.loc[:,'plant_id'].eq(current_plant['plant_id'])
        current_mappings = df_mapping.loc[slic_]
        current_id = current_plant['plant_id']
        artificial = current_mappings.loc[:,'artificial']
        temp_dict = {}
        temp_dict[value_column] = current_plant[value_column]
        temp_dict['images'] = current_mappings.loc[:,'img_name'].to_list()
        temp_dict['artificial_mask'] = artificial.to_list()
        plants[current_id] = temp_dict

    df_plants = pd.DataFrame.from_dict(plants, orient='index').sort_index()

    return df_plants

def split_genotype_(
        plant_mapping : pd.DataFrame,
        rel_test_set_size : float,
        rel_val_set_size : float,
        random_generator : np.random.Generator,
        verbose : bool = False,
        manual_geno_key : Dict[str, str] = {}
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to split the `plant_mapping` in training, validation and test mapping
    
    The validation & test only contain plants with at least one
    non artificial image (unless all images are artificial).
    Artificial images are deleted from the test and validation mapping.

    Args:
        plant_mapping (pd.DataFrame): plant_id -> volume, plant_id -> images mapping
        rel_test_set_size (float, optional): Relative (to number of plants) test set size. Defaults to 0.2.
        rel_val_set_size (float, optional): Relative (to number of plants) validation set size. Defaults to 0.1.
        verbose (bool, optional): Print split sizes. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_mapping, validation_mapping, test_mapping

    Warnings:
        Warns if the plant mapping contains less than 50 plants with real world images.
        
    Examples:
        >>> plant_mapping
                volume               images artificial_mask
        2_6_1   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_2   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_3   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_4   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_5   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_6   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_7   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_8   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_9   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_10  5000.0  [2_6_1_0.jpg, arti]   [False, True]
        >>> train, val, test = split_(plant_mapping=df_in,
        ...                           rel_test_set_size=0.2,
        ...                           rel_val_set_size=0.1,
        ...                           random_generator=np.random.default_rng(0))
        >>> val
               volume         images artificial_mask
        2_6_1  5000.0  [2_6_1_0.jpg]         [False]
    """

    #add genotype id to the mapping dataframe
    n_plants = plant_mapping.shape[0]
    plant_mapping['genotype_id'] = plant_mapping.index.to_series().astype(str).str.extract(r'^(\d+_\d+)')
    
    genotype_plant_map : Dict[str, List[int]] = {}
    for i in range(n_plants):
        genotype_id = plant_mapping.iloc[i]['genotype_id']
        if genotype_id in manual_geno_key:
            genotype_id = manual_geno_key[genotype_id]

        if genotype_id in genotype_plant_map:
            genotype_plant_map[genotype_id].append(i)
        else:
            genotype_plant_map[genotype_id] = [i]
        
    genotype_ids = np.array(list(genotype_plant_map.keys()))
    n_genos = genotype_ids.shape[0]
    shuffled_idx = np.arange(n_genos)
    random_generator.shuffle(shuffled_idx)

    # Get the split indices on all genotypes
    test_size = int(rel_test_set_size * n_genos)
    val_size = int(rel_val_set_size * n_genos)
    test_idx , val_idx, train_idx = np.split(shuffled_idx, [test_size, test_size + val_size])

    train_plant_ids = []
    val_plant_ids = []
    test_plant_ids = []

    for geno_id in genotype_ids[train_idx]:
        train_plant_ids += genotype_plant_map[geno_id]

    for geno_id in genotype_ids[val_idx]:
        val_plant_ids += genotype_plant_map[geno_id]
        
    for geno_id in genotype_ids[test_idx]:
        test_plant_ids += genotype_plant_map[geno_id]    

    train_mapping = plant_mapping.iloc[train_plant_ids]
    test_mapping = plant_mapping.iloc[test_plant_ids]
    val_mapping = plant_mapping.iloc[val_plant_ids]
    
    assert np.intersect1d(train_mapping.index, val_mapping.index).shape[0] == 0
    assert np.intersect1d(train_mapping.index, test_mapping.index).shape[0] == 0
    assert np.intersect1d(val_mapping.index, test_mapping.index).shape[0] == 0
    
    # remove the artificial images from test and val
    def remove_artificial(row : pd.Series):
        mask = np.array(row['artificial_mask'])
        images = np.array(row['images'])[~mask]
        mask = mask[~mask]
        row['artificial_mask'] = mask.tolist()
        row['images'] = images.tolist()
        return row
    val_mapping = val_mapping.apply(remove_artificial, axis=1)
    test_mapping = test_mapping.apply(remove_artificial, axis=1)
    
    if verbose:
        print(f"Plants in mapping: {n_plants}.")
        print(f"Total train set size {train_mapping.shape[0]}")
        print(f"Total validation set size {val_mapping.shape[0]}")
        print(f"Total test set size {test_mapping.shape[0]}")

    return train_mapping, val_mapping, test_mapping

def compute_and_store_split(
        df_mapping : pd.DataFrame, 
        mapping_plants_path : Path,
        mapping_train_path : Path,
        mapping_val_path : Path, 
        mapping_test_path : Path, 
        value_column : str,
        random_generator : np.random.Generator,
        rel_test_set_size : float = 0.2,
        rel_val_set_size : float = 0.1,
        verbose : bool = False,
        ):
    """Compute the train / val / test split and store the output
    in json format at the given locations.
    
    The output files will be overwritten if already exsitent.

    Args:
        mapping_image_path (DataFrame): Datafraem file containing the image to volume / plant mapping.
        mapping_plants_path (Path): json file path to store plant to volume / images mapping
        mapping_train_path (Path): json file to store subset of plants_mapping for training
        mapping_val_path (Path): json file to store subset of plants_mapping for validation
        mapping_test_path (Path): json file to store subset of plants_mapping for testing
        value_column (str): Column name containing the value.
        random_generator (Generator): Randomness to pull shuffling from.
        rel_test_set_size (float, optional): Relative test set size (to number of plants). Defaults to 0.2.
        rel_val_set_size (float, optional): Relative val set size (to number of plants). Defaults to 0.1.
        verbose (bool, optional): Defaults to False.
    
    Examples:
        >>> compute_and_store_split(df_mapping=df_mapping,
                                    mapping_plants_path = "data/plants_mapping.json",
                                    mapping_train_path = "data/train_mapping.json",
                                    mapping_val_path = "data/val_mapping.json",
                                    mapping_test_path = "data/test_mapping.json",
                                    value_column = "value",
                                    random_generator = np.random.default_rng(0),
                                    rel_test_set_size = 0.2,
                                    rel_val_set_size = 0.1,
                                    verbose=True)
    """
    
    plant_mapping = compute_plant_mapping_(df_mapping=df_mapping,
                                           value_column=value_column)

    
    #Design 2023
    design_2023 = pd.read_csv("data/Design_2023.csv", encoding="ISO-8859-1")
    design_2023['genotype_id'] = design_2023['range_lot'].astype(str) + "_" + design_2023['row_lot'].astype(str)
    design_2023 = design_2023[['genotype_id', 'genotype_name']]

    #Design 2023
    design_2023_2 = pd.read_csv("data/Design_small.csv", encoding="ISO-8859-1")
    design_2023_2['genotype_id'] = design_2023_2['range_lot'].astype(str) + "_" + design_2023_2['row_lot'].astype(str)
    design_2023_2 = design_2023_2[['genotype_id', 'genotype_name']]

    #Design 2024
    design_2024 = pd.read_csv("data/Design_2024.csv", encoding="ISO-8859-1")
    design_2024['genotype_id'] = design_2024['range_lot'].astype(str) + "_" + design_2024['row_lot'].astype(str)
    design_2024 = design_2024[['genotype_id', 'genotype_name']]
  
    #merge them
    design_2023_2024 = pd.concat([design_2023, design_2024], ignore_index=True)
    design_2023_2024 = design_2023_2024.drop_duplicates(subset=['genotype_id', 'genotype_name'], keep='last')
    design_2023_2024_J = pd.concat([design_2023_2024, design_2023_2], ignore_index=True)
    design_2023_2024_J = design_2023_2024_J.drop_duplicates(subset=['genotype_id', 'genotype_name'], keep='last')
    
    manual_geno_map = dict(zip(design_2023_2024_J['genotype_id'].astype(str), design_2023_2024_J['genotype_name'].astype(str)))

    # split the genotypes in train, test and validation set
    train_mapping, val_mapping, test_mapping = split_genotype_(plant_mapping=plant_mapping,
                                                      rel_test_set_size=rel_test_set_size,
                                                      rel_val_set_size=rel_val_set_size,
                                                      random_generator=random_generator,
                                                      manual_geno_key=manual_geno_map,
                                                      verbose=verbose)

    mapping_train_path.parent.mkdir(parents=False, exist_ok=True)
    train_mapping.to_json(mapping_train_path, orient='index')

    mapping_val_path.parent.mkdir(parents=False, exist_ok=True)
    val_mapping.to_json(mapping_val_path, orient='index')
    
    mapping_test_path.parent.mkdir(parents=False, exist_ok=True)
    test_mapping.to_json(mapping_test_path, orient='index')
    
    mapping_plants_path.parent.mkdir(parents=False, exist_ok=True)
    plant_mapping.to_json(mapping_plants_path, orient='index')

