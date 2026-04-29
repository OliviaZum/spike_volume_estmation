from icecream import ic
import numpy as np
from typing import Optional, Tuple, List
import cv2
import numpy.typing as npt
from copy import deepcopy
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import interpolate
from pathlib import Path
from prep_lib.img_segmenter import ImageSegmenter, print_time
from prep_lib.abstract_spike import AbstractSpike
from datetime import datetime
import time 

class SpikeSegmenter(ImageSegmenter):

    def __init__(self, segmentation_model : Tuple[str, str], verbose : bool = False):
        """Class to Segment spike and create an abstract representation of spike

        Args:
            segmentation_model (Tuple[str,Path]): Tuple with segment_anything type and model checkpoint
            verbose (bool, optional): Print times for each step. Defaults to False.
        """
        
        # set function timer features
        ImageSegmenter.__init__(self, segmentation_model=segmentation_model, verbose=verbose) 
    
    def get_points(self, image : npt.ArrayLike, num_points = 1) -> npt.NDArray:
        """Compute points on spike, which have the right hue, value and s

        Image is assumed to be blurred and resized in the input already

        Args:
            image (npt.ArrayLike): Image containig bars on the left side centered

        Returns:
            npt.NDArray: two coordinates within the image
        """
        
        # convert to correct color scheme
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        #np.set_printoptions(threshold=np.inf)  # disables summarization
        #print(hsv_image)

        
        # hue in the range of spikes
        #bar removal
        #spike_mask = (hsv_image[:,:,0] <= 25) & (hsv_image[:,:,0] >= 10)

        #stalk removal
        spike_mask = (hsv_image[:,:,0] <= 25) & (hsv_image[:,:,0] >= 10) #orange pixels, background


        # select point with max value
        max_hue_val_sat = np.flip(np.sort(hsv_image[spike_mask], axis=0),axis=0)[num_points]
        spike_mask = spike_mask & (hsv_image[:,:,1] >= max_hue_val_sat[1])

        # select point with max saturation
        max_hue_val_sat = np.flip(np.sort(hsv_image[spike_mask], axis=0),axis=0)[num_points]
        spike_mask = spike_mask & (hsv_image[:,:,2] >= max_hue_val_sat[2])

        # get point coordinates
        spike_points = np.flip(np.argwhere(spike_mask),axis=1)
        
        if num_points == 1:
            spike_points = spike_points[[0]]
        else:
            spike_points = spike_points[:num_points]

        
        return spike_points
    

    # time is printed within segmentor
    def segment_spike(self, rgb_image : npt.NDArray, baseline: bool) -> npt.NDArray:
        """Computes spike from rgb image

        Args:
            rgb_image (npt.NDArray): image segmentation

        Returns:
            npt.NDArray: mask for the segmented spike
        """
        select_index = 1
        num_points = 1
       
        
        if baseline: 
            kernel_size = 13
            downsize = 2024 # note that kernel size is absolute, so samller downsize => larger relative kernel
        else: 
            kernel_size = 3
            downsize = 512 # note that kernel size is absolute, so samller downsize => larger relative kernel

        print(kernel_size)
        print(downsize)

        mask = self.segment_pipline(rgb_image=rgb_image,select_index=select_index, kernel_size=kernel_size, resize=downsize, num_points=num_points)
        
       
        # remove mask artefacts
        mask = mask.astype(np.uint8)

        if baseline: 
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5), (2,2))
            mask = cv2.erode(mask,element,iterations=8)
        else: 
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3), (1,1))
            mask = cv2.erode(mask,element,iterations=1)

        #save mask
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        #masksave = (mask * 255).astype(np.uint8)
        #cv2.imwrite(f"data/check_images/mask_output{timestamp}.png", masksave)
        
        mask = cv2.dilate(mask,element,iterations=1)
        #mask = (mask > 0).astype(np.uint8) * 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels >1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = 1 + np.argmax(areas)
            mask = (labels == largest_label).astype(np.uint8)

        #save mask
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        #masksave = (mask * 255).astype(np.uint8)
        #cv2.imwrite(f"data/check_images/last-mask{timestamp}.png", masksave)


        
        return mask
    
    def next_neighbours(self, padded_skeleton : npt.NDArray, point : npt.NDArray, max_d : int) ->  Optional[npt.NDArray]:
        """Compute neighbours of given point if there exists one
        
        NOTE: If the skeleton in padded_skeleton goes to up until the border of the image, algorithm might thow
        and out of bound error. Idea to fix this, is to add 0 padding. BUT this is the CALLERS BURDON for
        computational efficiency reasons.
        
        Args:
            padded_skeleton (npt.NDArray): skeleton with 0 on the outermost pixels of the array
            point (npt.NDArray): current point
            prev_neighbours (npt.NDArray): already visited points
            max_d (int): maximal distance to look for neighbours before returning None
        
        Returns:
            Optional[npt.NDArray]: List of neighbour points
        """
        x = point[0]
        y = point[1]

        # delta x, delta y
        d = 1
        neighbours = padded_skeleton[x-d:x+1+d, y-d:y+1+d]
        while np.all(~neighbours) and d < max_d:
            # allow for larger hops
            d = d + 1
            neighbours = padded_skeleton[x-d:x+1+d, y-d:y+1+d]
            
        num_neighbours = np.sum(neighbours)
        
        if num_neighbours == 0:
            # no next neighbour
            return None

        next_points_local = np.argwhere(neighbours.astype(bool))
        
        # compute global location
        next_point = point + next_points_local - (d,d)
        
        return next_point

    def unset_skeleton(self, skeleton : npt.NDArray, points : npt.NDArray, distance : int = 0) -> None:
        """Set points to zero on skeleton

        Args:
            skeleton (npt.NDArray): sekeleton to be updated
            points (npt.NDArray): points to set to zero
        """
        d = distance
        
        if points.ndim == 1:
            x, y = points
            skeleton[x-d : x+1+d, y-d : y+1+d] = False
        elif points.ndim == 2:
            for x,y in points:
                skeleton[x-d : x+1+d, y-d : y+1+d] = False

            
    def get_skeleton_path(self, padded_skeleton : npt.NDArray, first_point : npt.NDArray, max_d : int, initial_point : bool = False) -> List[npt.NDArray]:
        """Compute the longest possible path starting at the given point
        
        function recurses if there are mutliple options
        
        If inital points flat is given, 2 paths might be returned

        Args:
            padded_skeleton (npt.NDArray): padded skeleton in which to search for a longest path
            first_point (npt.NDArray): starting points to search for path
            max_d (int): maximal distance
            initial_point (bool): indicator if the point is initial point, might return two concated path if it is

        Returns:
            List[npt.NDArray]: list of points in order which give a path
        """

        self.unset_skeleton(skeleton=padded_skeleton, points=first_point)
        path = [first_point]
        
        # get next points
        next_points = self.next_neighbours(padded_skeleton=padded_skeleton, point=first_point, max_d=max_d)
        
        # loop to check for new paths
        while not next_points is None:

            if next_points.shape[0] > 1:
                # recurse to get longest path
                self.unset_skeleton(skeleton=padded_skeleton, points=next_points, distance=max_d-1)
                subpaths = []
                for point in next_points:
                    subpaths.append(self.get_skeleton_path(padded_skeleton=padded_skeleton, first_point=point, max_d=max_d))
                
                # sort descending
                subpaths = sorted(subpaths, key=lambda path : -len(path))
                if initial_point:
                    second_subpath = subpaths[1]
                    second_subpath.pop(0)
                    second_subpath = list(reversed(second_subpath))
                    path = second_subpath + subpaths[0]
                else:
                    path = path + subpaths[0]
                return path
                
            else:
                # only one single neighbour found
                initial_point = False
                next_point = next_points[0]
                path.append(next_point)
                self.unset_skeleton(skeleton=padded_skeleton, points=next_point)
                next_points = self.next_neighbours(padded_skeleton=padded_skeleton, point=next_point, max_d=max_d)
        
        return path

    def save_axis(self,image : npt.NDArray, dir : Path, img_id : str):
        """If the image was segmented beforehand and abstract spike was
        created, then this function saves the main_curve, main_axis and
        skeleton into the given folder.
        
        This function is mainly for debugging, documentation and visualization
        it does not have any functional purpose apart from this.

        Args:
            image (npt.NDArray): image to plot onto
            dir (Path): directory where to put the files
            img_id (str): File names
        """
        dir.mkdir(parents=True, exist_ok=True)
        image = deepcopy(image)
        dpi = 96
        img_shape = tuple(i for i in np.flip(image.shape[:2]) // dpi)
        color = "red"
        linewidth = 10
        if hasattr(self, "main_curve") and not self.main_curve is None:
            print(f"Saving spline image of {img_id} at {dir}")
            plt.clf()
            plt.figure(figsize=img_shape, frameon=False)
            plt.axis("off")
            plt.imshow(image)
            plt.plot(self.main_curve[:,1], self.main_curve[:,0], color=color, linewidth=linewidth)
            plt.tight_layout(pad=0)
            plt.savefig(dir / f'{img_id}_curve.jpg', format='jpg', dpi=dpi)
        if hasattr(self, "main_axis") and not self.main_axis is None:
            print(f"Saving main axis of {img_id} at {dir}")
            plt.clf()
            plt.figure(figsize=img_shape, frameon=False)
            plt.axis("off")
            plt.imshow(image)
            plt.plot(self.main_axis[:,1], self.main_axis[:,0], color=color, linewidth=linewidth)
            plt.tight_layout(pad=0)
            plt.savefig(dir / f'{img_id}_axis.jpg', format='jpg',dpi=dpi)
        if hasattr(self, "width_pairs") and not self.width_pairs is None and hasattr(self, "main_curve") and not self.main_curve is None:
            print(f"Saving width paris of {img_id} at {dir}")
            plt.clf()
            plt.figure(figsize=img_shape, frameon=False)
            plt.axis("off")
            plt.imshow(image)
            for pair in self.width_pairs:
                plt.plot(pair[:,1], pair[:,0], color=color, linewidth=linewidth)
            plt.plot(self.main_curve[:,1], self.main_curve[:,0], color=color, linewidth=linewidth)
            plt.tight_layout(pad=0)
            plt.savefig(dir / f'{img_id}_widths.jpg', format='jpg',dpi=dpi)

        if hasattr(self, "base_skeleton") and not self.base_skeleton is None:
            print(f"Saving skeleton of {img_id}")
            image[self.base_skeleton.astype(bool)] = [255,0,0]
            plt.clf()
            plt.imsave(dir / f'{img_id}_skeleton.jpg', image)

    @print_time("initialize abstract spline")
    def initialize_abstract_spline(self, np_mask_contour : npt.NDArray) -> AbstractSpike:
        """Generate abstract spline, by creating the skeleton of the mask

        Args:
            np_mask_contour (npt.NDArray): mask of spline contours in the image

        Returns:
            AbstractSpike: abstract representation of the spike
        """
        
        
        # store original mask shape
        shape = np.array(np_mask_contour.shape)

        # make image smaller
        # note that for larger downsize values, it is adisable to use a larger max_d otherwise
        # jumps small jumps might not be recognized as connected component
        max_d = 5
        downsize = 500
        np_mask = cv2.resize(np_mask_contour.astype(np.uint8), (downsize,downsize))

        # compute skeleton
        skeleton = morphology.skeletonize(np_mask.astype(bool))
        self.base_skeleton = cv2.resize(skeleton.astype(np.uint8) * 255, np.flip(shape))
        # pad skeleton to make sure get_skeleton_path() works as intended
        padded_skeleton = np.pad(array=skeleton, pad_width=max_d, mode='constant')
        # pick any point and start building
        points = np.argwhere(padded_skeleton)
        first_point = points[0]

        
        ###########################################
        # Get order on the points
        ###########################################

        # start one direction
        points = self.get_skeleton_path(padded_skeleton=padded_skeleton, first_point=first_point, max_d=max_d, initial_point=True)
        points = np.array(points)

        ########################################################
        # transform the points back to the original coordinates
        ########################################################
        points = points - (max_d, max_d) # account for padding
        points[:,0] = points[:,0] * shape[0] // downsize + shape[0] // (downsize)  # account for rounding
        points[:,1] = points[:,1] * shape[1] // downsize + shape[1] // (downsize) # account for rounding

        # store for possible plotting
        self.main_axis = deepcopy(points)
        
        # get x and y spline
        lam = 0.00001 # smooting parameter
        time = np.linspace(0,1, points.shape[0])
        x_spline = interpolate.make_smoothing_spline(time, points[:,0], lam = lam)
        y_spline = interpolate.make_smoothing_spline(time, points[:,1], lam = lam)


        # create spike
        spike = AbstractSpike(x_spline=x_spline, y_spline=y_spline, contour_mask=np_mask_contour, verbose=self.verbose)
        # compute orthogonal vectors for visualization
        self.width_pairs = []
        for t in np.linspace(0,1,600):
            _, (left,_, right) = spike.radius(t)
            self.width_pairs.append(np.array([left, right]))
            
        
        # store for possible plotting
        self.main_curve = np.array([spike.spline(t) for t in np.linspace(0,1,600)])

        return spike
    
    def remove_stalk_from_mask(self,np_mask_contour : npt.NDArray, save_at : Path) -> npt.NDArray:
        """Generate a refined maks of spike, by creating the skeleton of the mask and by
        removing the thin stalk part 

        Args:
            np_mask_contour (npt.NDArray): mask of spline contours in the image
            save_at (Path): Path to save image for stalk removal

        Returns:
            refined_maks: maks of spike with removed stalk
        """
        
        # store original mask shape
        shape = np.array(np_mask_contour.shape)

        # make image smaller
        # note that for larger downsize values, it is adisable to use a larger max_d otherwise
        # jumps small jumps might not be recognized as connected component
        max_d = 5
        downsize = 500
        np_mask = cv2.resize(np_mask_contour.astype(np.uint8), (downsize,downsize))

        # compute skeleton
        skeleton = morphology.skeletonize(np_mask.astype(bool))
        self.base_skeleton = cv2.resize(skeleton.astype(np.uint8) * 255, np.flip(shape))
        # pad skeleton to make sure get_skeleton_path() works as intended
        padded_skeleton = np.pad(array=skeleton, pad_width=max_d, mode='constant')
        # pick any point and start building
        points = np.argwhere(padded_skeleton)
        first_point = points[0]
       
        ###########################################
        # Get order on the points
        ###########################################

        # start one direction
        points = self.get_skeleton_path(padded_skeleton=padded_skeleton, first_point=first_point, max_d=max_d, initial_point=True)
        points = np.array(points)
        
        

        ########################################################
        # transform the points back to the original coordinates
        ########################################################
        points = points - (max_d, max_d) # account for padding
        points[:,0] = points[:,0] * shape[0] // downsize + shape[0] // (downsize)  # account for rounding
        points[:,1] = points[:,1] * shape[1] // downsize + shape[1] // (downsize) # account for rounding

        # store for possible plotting
        self.main_axis = deepcopy(points)
        
        # get x and y spline
        lam = 0.00001 # smooting parameter
        time = np.linspace(0,1, points.shape[0])
        x_spline = interpolate.make_smoothing_spline(time, points[:,0], lam = lam)
        y_spline = interpolate.make_smoothing_spline(time, points[:,1], lam = lam)
        
    
        # create spike
        spike = AbstractSpike(x_spline=x_spline, y_spline=y_spline, contour_mask=np_mask_contour, verbose=self.verbose)
        # compute orthogonal vectors for visualization
        self.width_pairs = []
        widths = []
        
        for t in np.linspace(0,1,200):
            radius, (left,_, right) = spike.radius(t)
            
            self.width_pairs.append(np.array([left, right]))
            widths.append(radius) 
            
        # store for possible plotting
        self.main_curve = np.array([spike.spline(t) for t in np.linspace(0,1,200)])
            
    
        # Step 8: Remove parts of the mask where width is below threshold
        refined_mask = np_mask_contour.copy()  # Use your existing mask variable

        
        sorted_indices = np.arange(len(widths))
        sorted_widths = np.array(widths)
       
        sorted_widths = sorted_widths[::-1]
        sorted_indices = sorted_indices[::-1]


        # Step 2: Remove points from bottom to top until the width threshold is met
        t_values = np.linspace(0, 1, len(widths))
        last_thin_t = None
        max_width = 0

        # Define the percentage increase threshold
        increase_threshold = 1.04# % increase

        # Start checking width increases
        for i in range(30, len(sorted_widths)):  # Start from the second element (i=1) to compare with previous
            #print("values")
            #print(sorted_widths[i])
            #print(sorted_widths[i-1])
            if sorted_widths[i] >= sorted_widths[i - 1] * increase_threshold:  
                last_thin_t = t_values[i]  # Store the t-value of this significant increase
                last_thin_t = 1 - last_thin_t  # Adjust for correct direction
                max_width = sorted_widths[i] * 2  
                break  # Stop once we find the first significant increase


        #print(f"Last thin position along the spline (t): {last_thin_t}")
        #print(f"Last thin position along spline: {last_thin_t}, Max stalk width: {max_width}")
        
        # Step 2: Find the four corner points of the single large removal box

        # Get the normal direction (perpendicular to the spline) at t=0 and t=last_thin_t
        normal_0 = spike.grad_perp(1) / np.linalg.norm(spike.grad_perp(1))
        normal_last = spike.grad_perp(last_thin_t) / np.linalg.norm(spike.grad_perp(last_thin_t))

        # Get the middle point of the plant at t=0 and t=last_thin_t
        _, (_, middle_0, _) = spike.radius(1)
        _, (_, middle_last, _) = spike.radius(last_thin_t)
        
        

        # Compute the four corners of the box
        half_width = max_width / 2
        
        # Get tangent direction at t=0 (along the yellow line)
        tangent_0 = spike.grad(1) / np.linalg.norm(spike.grad(1))  # Normalize to get unit vector

        # Extend `middle_0` in the direction of the spline
        extension_factor = 100  # Adjust this to control how much extra coverage is added
        middle_0 = middle_0 + (tangent_0 * extension_factor)  # Move in the spline direction


        box_left_top = middle_0 - (normal_0 * half_width * 5)  # Top-left corner
        box_right_top = middle_0 + (normal_0 * half_width * 5)  # Top-right corner
        box_left_bottom = middle_last - (normal_last * half_width * 5)  # Bottom-left corner
        box_right_bottom = middle_last + (normal_last * half_width * 5)  # Bottom-right corner

        # Convert to integer for mask manipulation
        box = np.array([
            box_left_top.astype(int),
            box_right_top.astype(int),
            box_right_bottom.astype(int),
            box_left_bottom.astype(int)
        ])

        # Ensure all points are within image bounds
        #box = np.clip(box, 0, [refined_mask.shape[1] - 1, refined_mask.shape[0] - 1])

        # Step 4: Convert the refined mask to black and white
        bw_mask = (refined_mask * 255).astype(np.uint8)  # Convert boolean mask to grayscale (0-255)

        # Step 5: Convert grayscale image to 3-channel for adding the red box
        bw_mask_with_box = cv2.cvtColor(bw_mask, cv2.COLOR_GRAY2BGR)  



        # Step 6: Draw the red box directly on the image
        box_fixed = np.array([[p[1], p[0]] for p in box]).astype(int)  # Swap (Y, X) → (X, Y)
        cv2.polylines(bw_mask_with_box, [box_fixed], isClosed=True, color=(0, 0, 255), thickness=12)  # Red box


        # Step 6: Draw middle points in **blue**
        middle_0 = middle_0[::-1].astype(int)  # Swap (Y, X) → (X, Y)
        middle_last = middle_last[::-1].astype(int)  # Swap (Y, X) → (X, Y)
        cv2.circle(bw_mask_with_box, tuple(middle_0), radius=5, color=(255, 0, 0), thickness=-1)  # Blue
        cv2.circle(bw_mask_with_box, tuple(middle_last), radius=5, color=(255, 0, 0), thickness=-1)  # Blue

        # Add text labels to indicate which point is t=0 and which is t=last_thin_t
        cv2.putText(bw_mask_with_box, "t=0", (middle_0[0] + 10, middle_0[1] + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.putText(bw_mask_with_box, "t=last_thin_t", (middle_last[0] + 10, middle_last[1] + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Fix box corners: Convert to (X, Y) format
        corner_points_fixed = np.array([[p[1], p[0]] for p in [box_left_top, box_right_top, box_left_bottom, box_right_bottom]]).astype(int)

        # Draw fixed green corner points (NO TEXT, ONLY POINTS!)
        for corner in corner_points_fixed:
            cv2.circle(bw_mask_with_box, tuple(corner), radius=15, color=(0, 255, 0), thickness=-1)  # Green

        # Swap (Y, X) to (X, Y) before drawing
        main_curve_fixed = np.array([[p[1], p[0]] for p in self.main_curve]).astype(int)

        for i in range(len(main_curve_fixed) - 1):
            pt1 = tuple(main_curve_fixed[i])
            pt2 = tuple(main_curve_fixed[i + 1])
            cv2.line(bw_mask_with_box, pt1, pt2, (255, 0, 0), thickness=12)

        # Step 7: Save the final mask with the red box as a JPG (no saving and reading again!)
        file_path_mask_box = str(Path(save_at) / "box_mask.jpg")
        #print(file_path_mask_box)

        #cv2.imwrite(file_path_mask_box, bw_mask_with_box, [cv2.IMWRITE_JPEG_QUALITY, 100])


        # Fill the box region with 0s (this defines the area to remove)
        # Set all pixels inside the box to 0 in the existing refined_mask
        cv2.fillPoly(refined_mask, [box_fixed], 0)
        # Save the updated refined mask as a black-and-white image
        
        #file_path_mask = str(Path(save_at) / "refined_mask.jpg")
        file_path_mask = str(Path("data/check") / "refined_mask.jpg")

        #print(file_path_mask)
        cv2.imwrite(file_path_mask, (refined_mask * 255).astype(np.uint8))

        return refined_mask 
    
    def keep_stalk_in_mask(self,np_mask_contour : npt.NDArray, save_at : Path, timestamp) -> npt.NDArray:
        """Generate a refined maks of spike, by creating the skeleton of the mask and by
        removing the thin stalk part 

        Args:
            np_mask_contour (npt.NDArray): mask of spline contours in the image
            save_at (Path): Path to save image for stalk removal

        Returns:
            refined_maks: maks of spike with removed stalk
        """
        
        # store original mask shape
        shape = np.array(np_mask_contour.shape)

        # make image smaller
        # note that for larger downsize values, it is adisable to use a larger max_d otherwise
        # jumps small jumps might not be recognized as connected component
        max_d = 5
        downsize = 500
        np_mask = cv2.resize(np_mask_contour.astype(np.uint8), (downsize,downsize))

        # compute skeleton
        skeleton = morphology.skeletonize(np_mask.astype(bool))
        self.base_skeleton = cv2.resize(skeleton.astype(np.uint8) * 255, np.flip(shape))
        # pad skeleton to make sure get_skeleton_path() works as intended
        padded_skeleton = np.pad(array=skeleton, pad_width=max_d, mode='constant')
        # pick any point and start building
        points = np.argwhere(padded_skeleton)
        first_point = points[0]
       
        ###########################################
        # Get order on the points
        ###########################################

        # start one direction
        points = self.get_skeleton_path(padded_skeleton=padded_skeleton, first_point=first_point, max_d=max_d, initial_point=True)
        points = np.array(points)
        
        ########################################################
        # transform the points back to the original coordinates
        ########################################################
        points = points - (max_d, max_d) # account for padding
        points[:,0] = points[:,0] * shape[0] // downsize + shape[0] // (downsize)  # account for rounding
        points[:,1] = points[:,1] * shape[1] // downsize + shape[1] // (downsize) # account for rounding

        # store for possible plotting
        self.main_axis = deepcopy(points)
        
        # get x and y spline
        lam = 0.00001 # smooting parameter
        time = np.linspace(0,1, points.shape[0])
        x_spline = interpolate.make_smoothing_spline(time, points[:,0], lam = lam)
        y_spline = interpolate.make_smoothing_spline(time, points[:,1], lam = lam)
        
    
        # create spike
        spike = AbstractSpike(x_spline=x_spline, y_spline=y_spline, contour_mask=np_mask_contour, verbose=self.verbose)
        # compute orthogonal vectors for visualization
        self.width_pairs = []
        widths = []
        
        for t in np.linspace(0,1,200):
            radius, (left,_, right) = spike.radius(t)
            
            self.width_pairs.append(np.array([left, right]))
            widths.append(radius) 
            
        # store for possible plotting
        self.main_curve = np.array([spike.spline(t) for t in np.linspace(0,1,200)])
            
    
        # Step 8: Remove parts of the mask where width is below threshold
        refined_mask = np_mask_contour.copy()  # Use your existing mask variable
        sorted_indices = np.arange(len(widths))
        sorted_widths = np.array(widths)
       
        sorted_widths = sorted_widths[::-1]
        sorted_indices = sorted_indices[::-1]


        # Step 2: Remove points from bottom to top until the width threshold is met
        t_values = np.linspace(0, 1, len(widths))
        last_thin_t = None
        max_width = 0

        # Define the percentage increase threshold
        increase_threshold = 1.04# % increase

        # Start checking width increases
        for i in range(30, len(sorted_widths)):  # Start from the second element (i=1) to compare with previous
            #print("values")
            #print(sorted_widths[i])
            #print(sorted_widths[i-1])
            if sorted_widths[i] >= sorted_widths[i - 1] * increase_threshold:  
                last_thin_t = t_values[i]  # Store the t-value of this significant increase
                last_thin_t = 1 - last_thin_t  # Adjust for correct direction
                max_width = sorted_widths[i] * 2  
                break  # Stop once we find the first significant increase


        #print(f"Last thin position along the spline (t): {last_thin_t}")
        #print(f"Last thin position along spline: {last_thin_t}, Max stalk width: {max_width}")
        
        # Step 2: Find the four corner points of the single large removal box

        # Get the normal direction (perpendicular to the spline) at t=0 and t=last_thin_t
        normal_0 = spike.grad_perp(1) / np.linalg.norm(spike.grad_perp(1))
        normal_last = spike.grad_perp(last_thin_t) / np.linalg.norm(spike.grad_perp(last_thin_t))

        # Get the middle point of the plant at t=0 and t=last_thin_t
        _, (_, middle_0, _) = spike.radius(1)
        _, (_, middle_last, _) = spike.radius(last_thin_t)
    
    
        # Compute the four corners of the box
        half_width = max_width / 2
        
        # Get tangent direction at t=0 (along the yellow line)
        tangent_0 = spike.grad(1) / np.linalg.norm(spike.grad(1))  # Normalize to get unit vector

        # Extend `middle_0` in the direction of the spline
        extension_factor = 50  # Adjust this to control how much extra coverage is added
        middle_0 = middle_0 + (tangent_0 * extension_factor)  # Move in the spline direction


        box_left_top = middle_0 - (normal_0 * half_width * 4)  # Top-left corner
        box_right_top = middle_0 + (normal_0 * half_width * 4)  # Top-right corner
        box_left_bottom = middle_last - (normal_last * half_width * 4)  # Bottom-left corner
        box_right_bottom = middle_last + (normal_last * half_width * 4)  # Bottom-right corner

        # Convert to integer for mask manipulation
        box = np.array([
            box_left_top.astype(int),
            box_right_top.astype(int),
            box_right_bottom.astype(int),
            box_left_bottom.astype(int)
        ])

        print("box points")
        print(box)

        # Ensure all points are within image bounds
        #box = np.clip(box, 0, [refined_mask.shape[1] - 1, refined_mask.shape[0] - 1])

        # Step 4: Convert the refined mask to black and white
        bw_mask = (refined_mask * 255).astype(np.uint8)  # Convert boolean mask to grayscale (0-255)

        # Step 5: Convert grayscale image to 3-channel for adding the red box
        bw_mask_with_box = cv2.cvtColor(bw_mask, cv2.COLOR_GRAY2BGR)  



        # Step 6: Draw the red box directly on the image
        box_fixed = np.array([[p[1], p[0]] for p in box]).astype(int)  # Swap (Y, X) → (X, Y)
        cv2.polylines(bw_mask_with_box, [box_fixed], isClosed=True, color=(0, 0, 255), thickness=2)  # Red box


        # Step 6: Draw middle points in **blue**
        middle_0 = middle_0[::-1].astype(int)  # Swap (Y, X) → (X, Y)
        middle_last = middle_last[::-1].astype(int)  # Swap (Y, X) → (X, Y)
        cv2.circle(bw_mask_with_box, tuple(middle_0), radius=5, color=(255, 0, 0), thickness=-1)  # Blue
        cv2.circle(bw_mask_with_box, tuple(middle_last), radius=5, color=(255, 0, 0), thickness=-1)  # Blue

        # Add text labels to indicate which point is t=0 and which is t=last_thin_t
        cv2.putText(bw_mask_with_box, "t=0", (middle_0[0] + 10, middle_0[1] + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.putText(bw_mask_with_box, "t=last_thin_t", (middle_last[0] + 10, middle_last[1] + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Fix box corners: Convert to (X, Y) format
        corner_points_fixed = np.array([[p[1], p[0]] for p in [box_left_top, box_right_top, box_left_bottom, box_right_bottom]]).astype(int)

        # Draw fixed green corner points (NO TEXT, ONLY POINTS!)
        for corner in corner_points_fixed:
            cv2.circle(bw_mask_with_box, tuple(corner), radius=5, color=(0, 255, 0), thickness=-1)  # Green

        # Swap (Y, X) to (X, Y) before drawing
        main_curve_fixed = np.array([[p[1], p[0]] for p in self.main_curve]).astype(int)

        for i in range(len(main_curve_fixed) - 1):
            pt1 = tuple(main_curve_fixed[i])
            pt2 = tuple(main_curve_fixed[i + 1])
            cv2.line(bw_mask_with_box, pt1, pt2, (0, 255, 255), thickness=1)

        # Step 7: Save the final mask with the red box as a JPG (no saving and reading again!)
        file_path_mask_box = str(Path(save_at) / "box_mask")
        print(file_path_mask_box)

        #cv2.imwrite(f'{file_path_mask_box}_{timestamp}.png', bw_mask_with_box, [cv2.IMWRITE_JPEG_QUALITY, 100])

        #get a copy of the mask and set all values to zero
        np_mask_cleaned = np_mask_contour.copy()
        np_mask_cleaned[:] = 0

        
        # Fill the box region with 1s (this defines the area to inpaint)
        # Set all pixels inside the box to 1 in the existing refined_mask
        cv2.fillPoly(np_mask_cleaned, [box_fixed], 1)
        # Save the updated  mask as a black-and-white image
    
        cv2.imwrite(f"data/check_images/refined-mask{timestamp}.png", (np_mask_cleaned * 255).astype(np.uint8))

        return np_mask_cleaned
        
        
    
    def segment_and_initialize_spike(self, rgb_image : npt.NDArray, save_at : Path) -> AbstractSpike:
        """First extracts spike form image and then initializes abstract spline

        Args:
            rgb_image (npt.NDArray): Image of a single spline
            save_at (Patch): Path to save images for stalk removal

        Returns:
            AbstractSpike: Abstract representation of spline
        """
        
        # compute mask
        print("now segmentation")
        start_segmentation = time.time()
        contour = self.segment_spike(rgb_image=rgb_image, baseline=True) #returns binary mask
        time_segmentation = time.time() - start_segmentation
        print(f'\t{"Time Segmentation: ":7s}{time_segmentation:>6.4f}',end='')
        #remove stalk from contour mask

        #start_remove_mask = time.time()
        refined_contour = self.remove_stalk_from_mask(np_mask_contour=contour, save_at=save_at)
        print("removed peduncle")
        #time_remove_mask = time.time() - start_remove_mask
        #print(f'\t{"Time Stalk removal: ":7s}{time_remove_mask:>6.4f}',end='')
        
        #don't remove stalk 
        #spike = self.initialize_abstract_spline(np_mask_contour=contour)

        #remove stalk 
        spike = self.initialize_abstract_spline(np_mask_contour=contour)
      
        return spike
        

    def segment_stalk(self, rgb_image : npt.NDArray, save_at : Path) -> AbstractSpike:
        """First extracts spike form image and then initializes abstract spline

        Args:
            rgb_image (npt.NDArray): Image of a single spline
            save_at (Patch): Path to save images for stalk removal

        Returns:
            AbstractSpike: Abstract representation of spline
        """
        
        # compute mask
        print("now segmentation")
        contour = self.segment_spike(rgb_image=rgb_image, baseline = False)
        #print(contour.shape)
        #mask_bw = (contour * 255).astype(np.uint8)
        #cv2.imwrite("data/check_images/mask_output.png", mask_bw)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        masksave = (contour * 255).astype(np.uint8)
        cv2.imwrite(f"data/check_images/contour-mask{timestamp}.png", masksave)
        
        #mask without stalk
        stalk_contour = self.keep_stalk_in_mask(np_mask_contour=contour, save_at=save_at, timestamp=timestamp)
        
    
        return stalk_contour


