import cv2
import numpy as np
import torch as th
import numpy.typing as npt
from typing import Tuple, Optional
from deepfill_pytorch import Generator
from prep_lib.img_segmenter import ImageSegmenter, print_time
from icecream import ic
import scipy.signal as ss
import matplotlib.pyplot as plt

class BarRemover(ImageSegmenter):

    def __init__(self, segmentation_model : Tuple[str, str], deepfill_weights_path : Generator, verbose : bool = False):
        """Class to remove bars form image and inpaint (halluzinate)
        the content where the bars covered the image.

        Args:
            segmentation_model (Tuple[str,Path]): Tuple with segment_anything type and model checkpoint
            deepfill_weights_path (Path): Model Weights to inpaint the images.
            verbose (bool, optional): Print times for each step. Defaults to False.
        """
        
        # set function timer features
        ImageSegmenter.__init__(self, segmentation_model=segmentation_model, verbose=verbose) 

        use_cuda_if_available = True
        device = 'cuda:1' if th.cuda.is_available() and use_cuda_if_available else 'cpu'
        if verbose:
            print(f'Selected device: {device}.')
        self.device = th.device(device)

        self.deepfill_model = Generator(checkpoint=deepfill_weights_path, return_flow=True, device=self.device)
        print(deepfill_weights_path)

        self.transformed_image : npt.NDArray = None
        self.bars_mask : npt.NDArray = None
        
        
    @print_time(text="get darkest point")
    def get_points(self, image : npt.ArrayLike) -> npt.NDArray:
        """Compute the darkest points within the region of the two bars

        Args:
            image (npt.ArrayLike): Image containig bars on the left side centered

        Returns:
            npt.NDArray: two coordinates within the image
        """

        # get hsv image to better distinguish bars from background
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        self.hsv_point_image = hsv_image
        
        # locate bars in vertical direction (heuristically bars have lower value than background in hsv model)
        considered_width = int(image.shape[1] * 0.05)
        peak_prominence = 30 # threshhold must be between 0 and 255 as we have pixel values
        peak_distance = image.shape[1] * 0.05 # minimal distance between bars
        location_array = np.mean(image[:,:considered_width,2],axis=1)
        peaks, _ = ss.find_peaks(-location_array, height=None, prominence=peak_prominence, distance=peak_distance)
        x_coord = np.ones_like(peaks) * considered_width // 2
        points = np.stack([x_coord, peaks], axis=1)
        
        return points
    
    def remove_bars(self, rgb_image : npt.NDArray, bar_mask : Optional[npt.NDArray] = None, inpaint : bool = True, seed = 1141997):
        """Segment and inpaint (depending on self.inpaint_)
        
        If bar mask is given, segmentation step is skipped.
        If the class has inpaint = False in the arguments, then the inpaining step is 
        skipped
        
        Note that the seed is newly set on every call for reproducability even on splitted computations
        
        Args:
            rgb_image (npt.NDArray): image to be transformed
            bar_mask (npt.NDArray | None, optional): bar mask (should match segmentation output). Defaults to None.
            inpaint (bool, optional): Indicator whether or not to inpaint image

        Returns:
            BarRemove: self
        """
        # reset seed
        self.reset_seed(seed=seed)
        
        # get bars
        select_index = 0
        kernel_size = 10
        downsize = 1024

        if bar_mask is None:
            bar_mask = self.segment_pipline(rgb_image=rgb_image,select_index=select_index, kernel_size=kernel_size, resize=downsize)

            # prefer larger masks to cover everything, remove artefacts
            bar_mask = bar_mask.astype(np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5), (2,2))
            bar_mask = cv2.erode(bar_mask,element,iterations=8)
            bar_mask = cv2.dilate(bar_mask,element,iterations=23)
            
        print(inpaint)

        if inpaint:
            np_inpainted = self.inpaint(rgb_image, bar_mask)
        else:
            np_inpainted = None

        self.transformed_image = np_inpainted
        self.bars_mask = bar_mask
        
        return self
    
    @print_time(text="inpaining")
    def inpaint(self, np_image : npt.NDArray, np_mask : npt.NDArray) -> npt.NDArray:
        """Completes image, with removed mask

        Args:
            np_image (ndarray): image to be completed (w,h,3)
            np_mask (ndarray): mask where the image is to be completed (w,h)

        Returns:
            npt.NDArray: completed image
        """
        # Full sized images cannot be processed due to memory limits
        img_shape = np.flip(np_image.shape[:2])
        new_shape = (1024,1024)

        np_image = cv2.resize(np_image, new_shape)
        np_mask = cv2.resize(np_mask.astype(np.uint8), new_shape).astype(bool)

        # convert mask to pytorch
        th_mask = (np.array([np_mask]*4, dtype=np.float32) * 255)
        th_mask = th.from_numpy(th_mask).to(self.device)
        
        # convert image to pytorch
        th_img = th.from_numpy(np_image.astype(np.float32)) / 255
        th_img = th_img.permute(2,0,1).to(self.device)
        
        output = self.deepfill_model.infer(th_img, th_mask, return_vals=['inpainted'])
        output = cv2.resize(output.astype(np.uint8), dsize=img_shape)

        return output
    