import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import warnings
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
from pathlib import Path
from typing import List, Tuple
from torchvision.transforms.v2 import Compose, Resize, CenterCrop
from torchvision.io import read_image
from torchvision.transforms.v2 import (
    Compose, Resize, CenterCrop, Normalize, ToDtype
)
import torchvision.transforms as transforms

from torchvision.transforms.v2.functional import to_pil_image
from PIL import Image
from torchvision.transforms.v2 import ToImage
import torchvision.transforms.v2.functional as F

class PlantDataError(Exception):
    """Error meant to point at missing data"""
    pass

class ImageInferenceDataset(Dataset):
    
    def __init__(self,
                 image_dir : Path | str,
                 experiment : pd.DataFrame,
                 model_sequence_size : int,
                 transform : list,
                 dtype : torch.dtype = torch.float32,
                 ):
        """
        Image loader for inference, experiment file determening the exeriments
        
        All image names in the exeriment files (i.e. in a "image" list) must
        exist in the directory, othewise an error is thrown
        
        Downsizes images to 256 times 256 as resnet only works
        Images are. Herefore we first do a center crop
        so it is advantageous to place the desired object in the middle
        of the image. After the corping image gets rescaled.
        
        Args:
            image_dir (Path | str): image direcotry containing images
            experiment (DataFrame): list of plant_id and images to feed into network
            model_seqence_size (int): Seqence lenght per input accepted by the model.
            transform (list): Model specific transform settings which should include crop, normalize and resize.
            dtype (torch.dtype): Type of the tensor outputs
            
        Notes:
            Make sure to use a custom collocation function for the dataloader.
        
        Examples:
        >>> experiment
        >>> dataset = ImageInferenceDataset(image_dir = "data/images", experiment = expriment, model_seqence_len = 3)
        >>> def collate_fn(batch : List[Tuple[torch.Tensor, pd.Series, torch.Tensor]]):
                tensors, dict_tuple, masks = zip(*batch)
                tensors = torch.stack(tensors=tensors, dim=0)
                masks = torch.stack(tensors=masks, dim=0)
                return tensors, list(dict_tuple), masks
        >>> data_loader = DataLoader(dataset=dataset,
                batch_size=32,
                shuffle=False,
                pin_memory=True,
                collate_fn=collate_fn)
        """
        
        self.image_dir = Path(image_dir)
        self.experiment = experiment
        self.dtype = dtype
        self.model_sequence_size = model_sequence_size
        self.transform = transform
        
        # check experiment
        self.n_experiments = self.experiment.shape[0]
        
        assert image_dir.exists(), "Image directory does not exist."

    def __len__(self):
        return self.n_experiments
    
    def __getitem__(self, idx):
        
        element = self.experiment.iloc[idx]
        image_list = element['images']
        
        torch_image_list = []
        for image in image_list:
            
            # read image and transform to floating point type
            torch_image = read_image(str(self.image_dir / image))
            torch_image = (torch_image.type(self.dtype) / 255).clamp(0,1)
            
            # resize, normalize and turn
            transform = torch.nn.Sequential(*self.transform)
            torch_image = transform(torch_image)

            def save_tensor_as_image(tensor, path):
                # Undo normalization if needed here
                to_pil = transforms.ToPILImage()
                img = to_pil(tensor.cpu())  # Convert tensor to PIL
                img.save(path)

            #save_tensor_as_image(torch_image, "output.png")


            torch_image_list.append(torch_image)
            
        mask = torch.zeros(self.model_sequence_size,dtype=bool)
        mask[len(torch_image_list):] = True
        
        torch_images_ = torch.stack(torch_image_list,dim=0)
        torch_images = torch.zeros((self.model_sequence_size, *tuple(torch_images_.shape[1:])))
        torch_images[~mask] = torch_images_
        
        return torch_images, element, mask

class MultiImageTrainDataset(Dataset):

    def __init__(self,
                 image_dir : Path | str,
                 plant_mapping_path : Path | str,
                 transform: list,
                 value_column : str,
                 max_seq_len : int,
                 min_seq_len : int,
                 random_seq_len : bool = True,
                 random_choice : bool = False,
                 rotate : bool = True,
                 Y_SCALE : float = 1,
                 Y_SHIFT : float = 0,
                 random_generator : np.random.Generator = np.random.default_rng(10),
                 dtype : torch.dtype = torch.float32,):
        
        """Image loader with lazy image loading.
        
        For image to be recognized, it must exist in the directory
        under the name given in the plant_mapping file
        file in value_mapping_path.
        
        Downsizes images to 256 times 256 as resnet only works
        Images are. Herefore we first do a center crop
        so it is advantageous to place the desired object in the middle
        of the image. After the corping image gets rescaled.
        
        Randomly rotates the image if rotate true is given.
        This can be done for exaple in combination with n_duplicates
        to 'upsample' dataset.
        
        Args:
            image_dir (Path | str): image direcotry containing images
            plant_mapping (Dataframe): dataframe file with columns: value_column and images
            containing a list of images for the index, which is the plant_id
            value_column (str): Name of the column to be read for the value
            rotate (bool, optional): Flag indicating whether or not to rotate images. Defaults to False.
            transform (list): Model specific transform settings which should include crop, normalize and resize.
            dtype (dtype, optional): dtype of image. Defaults to float32.
        
        Raises:
            FileExistsError: If Image directory does not exists
        """

        image_dir = Path(image_dir)
        self.image_dir = Path(image_dir)
        plant_mapping_path = Path(plant_mapping_path)
        self.random_sequence_len = random_seq_len
        self.random_choice = random_choice
        self.rotate = rotate
        self.value_column = value_column
        self.dtype = dtype
        self.Y_STD = Y_SCALE
        self.Y_MEAN = Y_SHIFT
        self.generator = random_generator
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.min_seq_len = np.maximum(min_seq_len, 1)

        #Warning
        if self.min_seq_len > self.max_seq_len:
            raise IndexError(("Chosen min seqence lenght is larger than chosen "
                             "Max seqence lenght in MiltiImageTrainDataset."))
        
        # fix torch seed for image rotations
        torch.manual_seed(self.generator.integers(0,1000000))

        #read plant mapping
        plant_mapping = pd.read_json(plant_mapping_path, orient='index', convert_axes=False, dtype={"plant_id" : str})
        plant_mapping = plant_mapping.rename_axis("plant_id")
        self.plant_mapping = plant_mapping.reset_index()
        
        if not image_dir.exists() or not image_dir.is_dir():
            raise FileExistsError(f"Image directory, {str(image_dir)}, does not exist.")
        
    def __len__(self):
        return self.plant_mapping.shape[0]
    
    def _check_image_list(self, image_list : List[str]) -> List[str]:
        """Computes sublist wit existing images by checking every image in list

        Args:
            image_list (List[str]): List to be checked

        Returns:
            List[str]: List with all existing images from image_list
        """

        new_image_list : List[str] = []
        for img in image_list:
            if (self.image_dir / img).exists():
                new_image_list.append(img)
        
        return new_image_list

    def _get_image(self, plant_idx : int, image_idx : int) -> torch.Tensor:
        """Loads image and transforms it.

        Args:
            plant_idx (int): Chosen plant index.
            image_idx (int): Chosen image index.

        Raises:
            Index_Error: For indexes out of bounds the existing plants / images corresponding to the plant
            PlantDataError: No images are found in the image folder which correspond to the plant.
            
        Warnings:
            If the chosen image does not exist in image_dir the next existing image
            corresponding to the given plant_idx will be chosen

        Returns:
            torch.Tensor: Transformed image as torch tensor of size (3 x 256 x 256).
        """

        image_list : List[str] = self.plant_mapping.loc[plant_idx, 'images']

        if 0 > plant_idx or plant_idx >= self.__len__():
            raise IndexError(f"Plant index {image_idx} out of boundes [0, {self.__len__()}.")
        if 0 > image_idx or image_idx >= len(image_list):
            raise IndexError(f"Image index {image_idx} out of boundes [0, {len(image_list)}) for {image_list}.")

        image_name = image_list[image_idx]
        image_path = self.image_dir / image_name
  
        if not image_path.exists():
            # only consider existing images
            new_image_list = self._check_image_list(image_list)
            if len(new_image_list) == 0:
                raise PlantDataError(f"None of the images: {image_list} for plant {plant_idx} exists.")
        
            # update plant mapping such that it only contains existing images
            self.plant_mapping.at[plant_idx, 'images'] = new_image_list
            next_image_idx = image_idx % len(new_image_list)
            warning_string = (f"Image: {str(image_path)} does not exists.\n"
                            f'The image list, {image_list}, was replaced by {new_image_list}. '
                            f'From which the image {new_image_list[next_image_idx]} is selected.')
            warnings.warn(UserWarning(warning_string))
            
            # recurse on new plant mapping
            return self._get_image(plant_idx=plant_idx, image_idx=next_image_idx)
        
        # read image and transform to floating point type
        torch_image = read_image(str(image_path))
        torch_image = (torch_image.type(self.dtype) / 255).clamp(0,1)

        # resize, normalize and turn
        transform_list =[  
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ]
        if self.rotate:
            transform_list.append(v2.RandomRotation((0,360)))
        
        transform_list += self.transform
        transform = torch.nn.Sequential(*transform_list)
        torch_image = transform(torch_image)
        
        return torch_image
        
    
    def _get_image_sequence(self, plant_idx : int) -> npt.NDArray:
        """Generate a random image subset corresponding the plant

        Args:
            plant_idx (int): Index of the plant

        Returns:
            npt.NDArray: Array with plant indices
        """
  
        image_list = self.plant_mapping.loc[plant_idx, 'images']
        sorted_image_list = sorted(image_list) # images of spike sorted
        n_images = len(sorted_image_list)

        # make sure to choose a valid number of images
        max_seq_len = np.minimum(self.max_seq_len, n_images)
        min_seq_len = np.minimum(self.min_seq_len, max_seq_len)
        
        #Warnings
        if max_seq_len < self.max_seq_len:
            plant = self.plant_mapping.loc[plant_idx,'plant_id']
            warning_string = (f"Plant {plant} only has {max_seq_len} images but {self.max_seq_len} are requested. "
                           f"In consequence only {max_seq_len} images were selected.")
            warnings.warn(UserWarning(warning_string))

        if min_seq_len < self.min_seq_len:
            warning_string = (f"The chosen min sequence length of {self.min_seq_len} is "
                           f"larger than the available max seqence lenght {max_seq_len} items.")
            warnings.warn(UserWarning(warning_string))
            

        if self.random_sequence_len and min_seq_len < max_seq_len:
            # select random number of images
            num_out = self.generator.integers(min_seq_len,max_seq_len)
        else:
            # select fixed number of images
            num_out = max_seq_len

        if self.random_choice: 
            #choose random indices
            image_index_choice = self.generator.choice(np.arange(n_images), size=num_out, replace=False)
        else: 
            #choose the first few images indices
            image_index_choice = np.arange(num_out)

        # order images of plant to have consitency
        self.plant_mapping.at[plant_idx, 'images'] = sorted_image_list
      
        return image_index_choice
        

    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, float]:
        """Get a training seqence of images for one plant

        Args:
            idx (int): plant index

        Returns:
            Tuple[torch.Tensor, float, torch.Tensor]: torch image seqence (self.max_seq_len x 3 x 256 x 256),
                scaled and shifted output label, unused mask (self.max_seq_len) (true if seqence element is not used)
        """
        
        # chose image indices for the given plant index
        image_index_choice = self._get_image_sequence(idx)
        num_out = image_index_choice.shape[0]

        key_padding_mask = torch.zeros(self.max_seq_len, dtype=bool)
        key_padding_mask[num_out:] = True # mask all entries not to be considered

        # collect images and check (in _get_image(..)) if they exist
        torch_image_list = []
        for image_idx in image_index_choice:
            torch_image = self._get_image(plant_idx=idx, image_idx=image_idx)
            torch_image_list.append(torch_image)
           
        torch_images_ = torch.stack(torch_image_list,dim=0)
        torch_images = torch.zeros((self.max_seq_len,*tuple(torch_images_.shape[1:])),dtype=torch_images_.dtype)
        torch_images[:num_out] = torch_images_

        out_label = self.plant_mapping.loc[idx, self.value_column]
        out_label = (out_label - self.Y_MEAN) / self.Y_STD
    
        return torch_images, out_label, key_padding_mask