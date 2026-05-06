import numpy as np
import pandas as pd
import torch
import warnings

from contextlib import nullcontext
from icecream import ic
from lib.seed_setter import SeedSetter
from pathlib import Path
from data_lib.data_saver import Saver
from data_lib.data_sets import ImageInferenceDataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, List
import time 


class Inferrer(SeedSetter):
    
    def __init__(self, saved_model_path : Path | str,
                 output_name : str,
                 Y_STD : float, 
                 Y_MEAN : float,
                 pretrained_model : nn.Module,
                 verbose : bool = False,
                 uncertainty : bool = False 
                 ):
        """Initializes an inferrer for a model given under the saved_model_path

        Args:
            saved_model_path (Path | str): Path to the saved model
            output_name (str) : Name of the prediction column in the output csv file.
                If it is 'volume', scaling and mean shift will be applied
            pretrained_model (Module) : Pretrained model to generate embeddings.
                The output of the model is expected the be the embeddings.
            verbose (bool) : Defaults to False.

        Raises:
            FileNotFoundError: If file saved model doesn't exsit.
        """
        
        # Fix inferrer seeds (should not make a difference)
        SeedSetter.__init__(self)
        self.saved_model_path = Path(saved_model_path)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0') 
            if verbose:
                print("Compute with GPU")
        else:
            self.device = torch.device("cpu")
            if verbose:
                print("Compute with CPU")

        #self.device = torch.device("cpu")
        #if verbose:
        #    print("Compute with CPU")
                    

        self.model, _ = Saver.load_model(self.saved_model_path)
        self.model = self.model.to(self.device)
        self.output_name = output_name.lower()
        self.pretrained_model = pretrained_model
        self.verbose = verbose
        self.uncertainty = uncertainty
        self.Y_MEAN_VOL = Y_MEAN
        self.Y_STD_VOL = Y_STD
        
    
    def infer_experiment(self, dataset : ImageInferenceDataset) -> pd.DataFrame:
        """ Predict volume for each element in the specified experiment and
        returns the experiment with a added / overwritten `self.output_name` value for each experiment.
        
        Changes the experiment input

        Args:
            dataset (Dataset) : Dataset containing the data to be loaded, the get item method is supposed
                to return a pandas.Series including the plant_id and the image_names
        Raises:
            FileNotFoundError: If the image direcotry or some files in the experiments are not found

        """

        # Define a data loader for the dataset
        def collate_fn(batch : List[Tuple[torch.Tensor, pd.Series, torch.Tensor]]):
            tensors, dict_tuple, masks = zip(*batch)
            tensors = torch.stack(tensors=tensors, dim=0)
            masks = torch.stack(tensors=masks, dim=0)
            return tensors, list(dict_tuple), masks
            
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=64,
                                 shuffle=False,
                                 pin_memory=True,
                                 collate_fn=collate_fn)
        

        
        # switch to eval (because of dropout...)

        if self.pretrained_model is not None:
            self.pretrained_model.eval()
            self.pretrained_model = self.pretrained_model.to(self.device)
        self.model.eval()
        self.model = self.model.to(self.device)


        # Extract the embeddings
        values = []
        outputs = []
        if self.verbose:
            print(f"Infer {len(data_loader)} batches.")
  
        for i, (image_batch, element_batch, mask_batch) in tqdm(enumerate(data_loader), disable=~self.verbose):

            start_eval_time = time.time()

            # gradient computations in evaluation are pointless so just dont.
            with torch.no_grad():
                with nullcontext() if self.device.type == 'cpu' else torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
                    image_batch_on_device = image_batch[~mask_batch].to(self.device)  
                    print(self.device)

                    if self.pretrained_model is not None: 
                        # Apply pretrained model
                        feature_batch_on_device = self.pretrained_model(image_batch_on_device)
                        # Restore the collapsed seqence dimension and pad to have the same sequence lenght, using the mask
                        full_batch_size = (mask_batch.shape[0], mask_batch.shape[1],feature_batch_on_device.shape[1])
                        feature_batch_full = torch.zeros(full_batch_size, dtype=feature_batch_on_device.dtype, device=self.device)
                        feature_batch_full[~mask_batch] = feature_batch_on_device #batchsize x sequence length x # features of pretrained network
                        assert tuple(feature_batch_full.shape) == full_batch_size
                    else:
                        feature_batch_full = image_batch.to(self.device)
                        
                    # Perform inference
                    batch_values = self.model(feature_batch_full, mask_batch.to(self.device))

                batch_values = batch_values.cpu().numpy().flatten() * self.Y_STD_VOL + self.Y_MEAN_VOL

                values.append(batch_values)
                outputs = outputs + element_batch

                eval_time = time.time() - start_eval_time
                print(f'\t{"Evaluation time: ":7s}{eval_time:>6.4f}')
                
        values = np.concatenate(values, axis=0)
        outputs = pd.DataFrame(outputs)
        outputs[self.output_name] = values

        
        return outputs
    

class ExperimentCreator():
    
    def create_experiment(self, plant_mapping : pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("Implement your own experiment")


class SingleImageExperimentCreator(ExperimentCreator):
    """Creator for experiment / testing data with single images

    The class is, as all ExperimentCreator classes, used
    to create json experiment files, which consist of a list
    of dictonaries, containing the keys `plant_id` and 
    `images`. The `plant_id` value will contain a plant
    id, e.g. `10_10_1`, and `images` will be a list
    of images, e.g. `['10_10_1_0_b.jpg', '10_10_1_1_a.jpg']`
    to be used as input to the model during inference time.
    """
    
    def __init__(self, ignore_artificial : bool):
        """Store local variables

        The `ignore_artificial` variable indicates whether or not
        artificial images should be considered in the experiment.
        If False all artificial images will be neglected. Unless
        there are only artificial images in the dataset.

        Args:
            ignore_artificial (bool): Whether or not artificial images should be considered.
        """
        self.ignore_artificial = ignore_artificial
    
    def create_experiment(self, plant_mapping : pd.DataFrame,) -> pd.DataFrame:
        """Creates an experiment with single image experiments for each image in plant_mapping
        
        Artificial images are ignored if `self.ignore_artificial` is true unless
        all image in plant_mapping are artificial images.

        Args:
            plant_mapping (pd.DataFrame): Dataframe with one row per plant_id, with
            a column `plant_id` containing the plant id, another column `images` containing
            a list of image names depicting the corresponding plant, a column `artificial_mask`
            containing a list of boolean values indicating for each image whether it is
            an artificial image or not. The two list have the same lenght and the same order.

        Returns:
            pd.DataFrame: A Dataframe with one row per experiemnt, a column containing the
            `plant_id` and another column containing the `images` used as input for the experiment.
        """
        output_dict = []
        plant_mapping_exploded = plant_mapping.explode(['images', 'artificial_mask'])
        if self.ignore_artificial and (not plant_mapping_exploded.loc[:,"artificial_mask"].all()):
            plant_mapping_exploded = plant_mapping_exploded.iloc[~np.array(plant_mapping_exploded['artificial_mask'], dtype=bool)]

        for plant_id, row in plant_mapping_exploded.iterrows():
            temp_dict = {}
            temp_dict["plant_id"] = plant_id
            images = row["images"]
            temp_dict["images"] = [images]
            output_dict.append(temp_dict)
        
        output_dict = pd.DataFrame(output_dict)

        return output_dict
    
class MultiImageExperimentCreator(ExperimentCreator):
    """Creator for experiment / testing data with multiple images

    The class is, as all ExperimentCreator classes, used
    to create json experiment files, which consist of a list
    of dictonaries, containing the keys `plant_id` and 
    `images`. The `plant_id` value will contain a plant
    id, e.g. `10_10_1`, and `images` will be a list
    of images, e.g. `['10_10_1_0_b.jpg', '10_10_1_1_a.jpg']`
    to be used as input to the model during inference time.
    """
    
    def __init__(self, 
                 ignore_artificial : bool, 
                 eval_max_seq_len : int,
                 eval_min_seq_len : int,
                 eval_random_seq_len : bool = True,
                 random_choice : bool = False):
        """Store local variables

        The `ignore_artificial` variable indicates whether or not
        artificial images should be considered in the experiment.
        If False all artificial images will be neglected. Unless
        there are only artificial images in the dataset.

        Args:
            ignore_artificial (bool): Whether or not artificial images should be considered.
        """
        self.ignore_artificial = ignore_artificial
        self.eval_max_seq_len = eval_max_seq_len
        self.eval_min_seq_len = eval_min_seq_len
        self.eval_random_seq_len = eval_random_seq_len
        self.random_choice = random_choice
        
    
    def create_experiment(self, plant_mapping : pd.DataFrame,) -> pd.DataFrame:
        """Creates an experiment with single image experiments for each image in plant_mapping
        
        Artificial images are ignored if `self.ignore_artificial` is true unless
        all image in plant_mapping are artificial images.

        Args:
            plant_mapping (pd.DataFrame): Dataframe with one row per plant_id, with
            a column `plant_id` containing the plant id, another column `images` containing
            a list of image names depicting the corresponding plant, a column `artificial_mask`
            containing a list of boolean values indicating for each image whether it is
            an artificial image or not. The two list have the same lenght and the same order.

        Returns:
            pd.DataFrame: A Dataframe with one row per experiemnt, a column containing the
            `plant_id` and another column containing the `images` used as input for the experiment.
        """
        output_dict = []
                
        plant_mapping_exploded = plant_mapping.explode(['images', 'artificial_mask'])

        if self.ignore_artificial and (not plant_mapping_exploded.loc[:,"artificial_mask"].all()):
            plant_mapping_exploded = plant_mapping_exploded[plant_mapping_exploded['artificial_mask'] == False]

        for plant_id, row in plant_mapping_exploded.iterrows():
            temp_dict = {}
            temp_dict["plant_id"] = plant_id
            
            filtered_df = plant_mapping_exploded.loc[plant_id]
            filtered_df = pd.DataFrame([filtered_df]) if isinstance(filtered_df, pd.Series) else filtered_df

            # Extract the image names from the filtered DataFrame
            images = filtered_df['images'].tolist()
            images.sort(key=lambda x: [int(part) if part.isdigit() else part for part in x.replace('.jpg', '').split('_')])

            eval_max_seq_len = np.minimum(self.eval_max_seq_len, len(images))
            eval_min_seq_len = np.minimum(self.eval_min_seq_len, eval_max_seq_len)
            
            #Give a warning if to much plants are requested:
            if eval_max_seq_len < self.eval_max_seq_len:
                warnings.warn(f"Plant {plant_id} only has {eval_max_seq_len} images but {self.eval_max_seq_len} are requested.")
            
            #Give a warning if not all images of a plant are available
            if eval_min_seq_len < self.eval_min_seq_len:
                warnings.warn((f"Images of plant {plant_id} has min sequence length of  {self.eval_min_seq_len}"
                            f"and max seqence lenght of {eval_max_seq_len} are requested."))
            
            
            if self.eval_random_seq_len and eval_min_seq_len < eval_max_seq_len:
                #if random == True and min smaller than max: take a random integer between min and max-1
                num_out = np.random.randint(eval_min_seq_len,eval_max_seq_len)
            else:
                #otherwhise we take the max number as output number
                num_out = eval_max_seq_len
            
            if self.random_choice: 
                #now we choose random images: 
                images_choice = np.random.choice(images, size=num_out, replace=False)
            else: 
                #choose the first few images 
                images_choice = images[:num_out]
          
            temp_dict["images"] = images_choice
            output_dict.append(temp_dict)
            
        #each element of the list will be one row, with columnn plant_id and images as list
        output_dict = pd.DataFrame(output_dict)
        output_dict = output_dict.drop_duplicates(subset=['plant_id'])   
        print(output_dict)     

        return output_dict