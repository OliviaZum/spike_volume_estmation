from segment_anything import SamPredictor, sam_model_registry
from lib.function_timer import FunctionTimer, print_time
from abc import abstractmethod, ABC
from typing import Tuple
from icecream import ic
import numpy.typing as npt
import numpy as np
import cv2
from lib.seed_setter import SeedSetter

class ImageSegmenter(FunctionTimer, ABC, SeedSetter):
    
    def __init__(self, segmentation_model : Tuple[str,str], verbose : bool, seed : int = 1141997):
        """Segemeter class, sets seed for reproducablity

        Args:
            segmentation_model (Tuple[str,str]): segment anything model
            verbose (bool): verbosity
            seed (int, optional): Defaults to 1141997.
        """
        
        FunctionTimer.__init__(self, verbose)
        SeedSetter.__init__(self, seed)
        
        # define models
        sam_type = segmentation_model[0]
        sam_model = segmentation_model[1]
        sam = sam_model_registry[sam_type](checkpoint=sam_model)
        self.sam_predictor : SamPredictor = SamPredictor(sam)
        
    
    @abstractmethod
    def get_points(self, image : npt.NDArray, **kwargs) -> npt.NDArray:
        """Compute the point(s) on object to be segmented later
        
        The method can take additional arguments

        Args:
            image (npt.ArrayLike): Image containig the object

        Returns:
            npt.NDArray: point list on the object
        """
        raise NotImplementedError('get_points must be defined when using the ImageSegmenter Superclass')
    
        
    @print_time(text="segment")
    def segment(self, rgb_image : npt.NDArray, point_coords : npt.NDArray) -> npt.NDArray:
        """Compute segmentation using metas segment anything

        Args:
            rgb_image (npt.NDArray): image in rgb colors
            point_coords (npt.NDArray): point coordinates on the desired object

        Returns:
            npt.NDArray: masks for the object usually 3 in increasing order (more covered area)
        """
        self.sam_predictor.set_image(rgb_image)
        point_labels = np.ones(point_coords.shape[0])
        masks, _, _ = self.sam_predictor.predict(point_coords=np.array(point_coords), point_labels=point_labels)
        
        return masks
    
    def blur(self, image : npt.NDArray, kernel_size : int) -> npt.NDArray:
        """Blurs an image with gaussian kernel of desired size

        Args:
            image (npt.NDArray): Input image to be blurred
            kernel_size (int): kernel size (results in symmetric size)

        Returns:
            npt.NDArray: blurred image
        """
        size = 2 * kernel_size + 1
        blurred : npt.NDArray = cv2.GaussianBlur(image, (size, size), kernel_size, kernel_size)
        return blurred

    def segment_pipline(self, rgb_image : npt.NDArray, select_index : int, kernel_size : int, resize : int = 1024, **kwargs) -> npt.NDArray:
        """Downsize -> Blur -> Segment -> Upsize
        
        Keyword arguments different form the ones listed below
        are passed to the get_points function.
        
        Handling the initial seed the callers duty, the class provied basic seeding features but make sure to call it in
        a controlled order or similar to ensure reproducability.
        
        Args:
            rgb_image (npt.NDArray): Image
            select_index (int): Mask index chosen in the end
            kernel_size (int): Kernel size for blurring
            resize (int, optional): Downsize to this size. Defaults to 1024.

        Returns:
            mask (npt.NDArray): boolean mask, masking the object
        """
        
        # store size and resize
        size = rgb_image.shape[:2]
        resized_img = cv2.resize(rgb_image, (resize, resize))
        
        # smooth image
        blurred_img = self.blur(resized_img, kernel_size=kernel_size)
        
        # get points for object detection
        points = self.get_points(image=blurred_img, **kwargs)
        print("points for object detection")
        print(points)
        
        
        masks = self.segment(rgb_image=blurred_img, point_coords=points)

        self.segment_masks = masks
        self.segment_blurred = blurred_img
        self.segment_points = points

        # chose mask 
        mask : npt.NDArray = masks[select_index]
        
        # up size (note that cv2 and numpy have different axis order hence flip)
        mask = cv2.resize(mask.astype(np.uint8), np.flip(size))
        
        return mask.astype(bool)
        
        

