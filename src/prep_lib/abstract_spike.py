import cv2
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy import interpolate, integrate
import open3d as o3d
from typing import Tuple
from pathlib import Path
from icecream import ic
from lib.function_timer import FunctionTimer, print_time
import warnings

class AbstractSpike(FunctionTimer):
    
    def __init__(self, x_spline : interpolate.BSpline, y_spline : interpolate.BSpline, contour_mask : npt.NDArray, verbose : bool):
        """Abstract representation of a spike as a parametrized curve in a 2 dimensional space

        Args:
            x_spline (interpolate.BSpline): parametrized x(t) coordinate, t in [0,1]
            y_spline (interpolate.BSpline): parametrized y(t) coordinate, t in [0,1]
            contour_mask (npt.NDArray): Mask of the hull around the spike image
            verbose (bool): Indicates whether times should be printed
        """
        FunctionTimer.__init__(self,verbose=verbose)

        # store parametrization
        self.x_spline = x_spline
        self.y_spline = y_spline
        
        self.x_deriv : interpolate.BSpline = x_spline.derivative(1)
        self.y_deriv : interpolate.BSpline = y_spline.derivative(1)
        
        # point information of spline
        self.spline = lambda t : np.array([self.x_spline(t), self.y_spline(t)])
        
        # gradient functions for points on spline
        self.grad = lambda t : np.array([self.x_deriv(t), self.y_deriv(t)])
        self.grad_perp = lambda t : np.array([-self.y_deriv(t), self.x_deriv(t)])
        self.grad_norm = lambda t : np.linalg.norm(self.grad(t))
        
        self.contour_mask = contour_mask
        
        # prep mask get contours
        _, thresh = cv2.threshold(self.contour_mask.astype(np.uint8), 0.5,  1, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        for c in contours:
            n = c.shape[0]
            # pick largest
            if n > contour.shape[0]:
                contour = c

        # get all contour points (note open cv and np have fliped coordinates)
        contour = contour.reshape(-1,2)
        self.contour = np.flip(contour,axis=1)

        # romberg divmax
        self.divmax = 11
        warnings.filterwarnings("ignore")

        # precompute lenght for faster access later
        self.lenght_scale_one = None
        self.lenght_scale_one = self.lenght(1.0)

        return None

    def skeleton(self, t : float) -> npt.NDArray:
        """Returns position in image coordinate system

        Args:
            t (float): Time t \in [0,1]

        Returns:
            npt.NDArray: position in floating point coordinates (image coordinate system)
        """
        x = self.x_spline(t)
        y = self.y_spline(t)
        return np.array([x,y], dtype=np.float32)
        
    def lenght(self, scale : float) -> float:
        """Length of the spike in "reality" along the representing curve

        Args:
            scale (float): size of one pixel in "reality" (realworld / image ratio)

        Returns:
            float: length
        """

        if self.lenght_scale_one is None:
            # integrate using trapezoidal rule
            # higher order methods do not converge as the splines have lacking smoothness prperties
            # compute equidistance points and add up
            n_points = 600
            integration_points = np.linspace(0,1,n_points)
            grad = np.vectorize(self.grad_norm)
            lenght = np.apply_along_axis(func1d=grad, axis=0, arr = integration_points).mean()
            lenght = scale * lenght
        else:
            lenght = scale * self.lenght_scale_one
        return lenght
    
    def radius(self, t : float) -> Tuple[float, npt.NDArray]:
        """Compute radius of spike at point t on the skeleton

        Returns 0 if there are no points found in one of the corners after the transformation

        Args:
            t (float): point on skeleton t \in [0,1]

        Returns:
            Tuple[float, npt.NDArray]: radius and [left_point, middle_point, enpoints] of diameter line on spike
        """
        
        point = self.spline(t)
        normal : npt.NDArray = self.grad_perp(t)
        
        x = 0
        y = 1
        
        # get angle to x axis
        alpha_neg =  np.arctan(normal[x] / normal[y])
        alpha = -alpha_neg

        # compute rotation matrix
        rot = np.array([[np.cos(alpha), np.sin(alpha)],[-np.sin(alpha), np.cos(alpha)]])
        rot_neg = np.array([[np.cos(alpha_neg), np.sin(alpha_neg)],[-np.sin(alpha_neg), np.cos(alpha_neg)]])

        # shift to origin
        shift_contour = self.contour - point
        # rotate s.t. normal goeas along y = axis 1
        st_contour = np.matmul(shift_contour, rot.T) # st = shift transformed
        
        # # contour points
        n_pt = st_contour.shape[0]
        all_idx = np.arange(n_pt)
        
        ps = []
        for t_ in np.linspace(0,1):
            ps.append(self.spline(t_))
        
        # divide into sectors
        # x < 0 -> top
        # x >= 0 -> bottom
        # y < 0 -> left
        # y >= 0 -> right
        maximal_radius = 400 # this corresponds to approximately 20mm
        mask_top = st_contour[:, x] < 0
        mask_bottom = st_contour[:, x] >= 0
        mask_left = (st_contour[:,y] < 0) & (st_contour[:,y] > -maximal_radius)
        mask_right = (st_contour[:,y] >= 0) & (st_contour[:,y] < maximal_radius)



        # check if spline fit is good enough to b inside spike if not assume radius 0
        if np.sum(mask_left) < 0 or np.sum(mask_right) < 0:
            return 0, np.array([point]*3)
        if np.sum(mask_top) < 0 or np.sum(mask_top) < 0:
            return 0, np.array([point]*3)
        
        # compute crucial points and
        abs_st_contour_x = np.abs(st_contour[:, x])
        point_index = np.zeros((2,2), dtype=int)
        
        # top left
        mask = mask_top & mask_left
        if np.sum(mask) == 0: return 0, np.array([point]*3)
        point_index[0,0] = all_idx[mask][np.argmin(abs_st_contour_x[mask])]
        
        # top right
        mask = mask_top & mask_right
        if np.sum(mask) == 0: return 0, np.array([point]*3)
        point_index[0,1] = all_idx[mask][np.argmin(abs_st_contour_x[mask])]

        # bottom left
        mask = mask_bottom & mask_left
        if np.sum(mask) == 0: return 0, np.array([point]*3)
        point_index[1,0] = all_idx[mask][np.argmin(abs_st_contour_x[mask])]

        # bottom left
        mask = mask_bottom & mask_right
        if np.sum(mask) == 0: return 0, np.array([point]*3)
        point_index[1,1] = all_idx[mask][np.argmin(abs_st_contour_x[mask])]
        
        # get y value at intersection of y axis
        right_weights = np.array([abs_st_contour_x[point_index[0,1]], abs_st_contour_x[point_index[1,1]]])
        right_ys = np.array([st_contour[point_index[0,1], y], st_contour[point_index[1,1], y]])
        left_weights = np.array([abs_st_contour_x[point_index[0,0]], abs_st_contour_x[point_index[1,0]]])
        left_ys = np.array([st_contour[point_index[0,0], y], st_contour[point_index[1,0],y]])
        
        right_y = (right_ys[0] * right_weights[1] + right_ys[1] * right_weights[0]) / np.sum(right_weights)
        left_y = (left_ys[0] * left_weights[1] + left_ys[1] * left_weights[0]) / np.sum(left_weights)
        middle_y = 0 # alternative would be (right_y + left_y) / 2 # should be close to 0 if spline is good
        
        radius = (right_y - left_y) / 2
        
        points = np.array([[0, left_y],[0, middle_y],[0, right_y]])
        points = np.matmul(points, rot_neg.T) + point
        
        return radius, points

    def area(self, t : float, scale : float) -> float:
        """Compute area at skeleton position t

        Args:
            t (float): skeleton position t \in [0,1]
            scale (float): realworld in mm / image in px ratio

        Returns:
            float: area at time t
        """
        
        # compute radius and scale with realworld / image ratio
        radius, _ = self.radius(t)
        radius = radius * scale
        
        return (radius ** 2) * np.pi

    @print_time("computing volume")
    def volume(self, scale : float) -> float:
        """Compute volume of spike given the scale

        Args:
            scale (float): (realworld in mm) / (image in pixel) ratio

        Returns:
            float: approimated spike volume
        """

        # lenght of the curve
        T = self.lenght(scale)
        
        #  give area function at timepoints
        func = lambda t : self.radius(t)[0] * scale
        func = np.vectorize(func)
        
        # integrate using trapezoidal rule
        # higher order methods do not converge as the splines have lacking smoothness prperties
        # compute equidistance points and add up
        n_points = 600
        integration_points = np.linspace(0,1,n_points)
        radii = np.apply_along_axis(func1d=func, axis=0, arr = integration_points)
        areas = radii**2 * np.pi
        volume = areas.mean() * T
        
        return volume
    
    def circle(self, t : float,  n_points : int) -> Tuple[npt.NDArray, npt.NDArray, float]:
        """Computes point in a cicle around the middle point of the spline at time t

        Args:
            t (float): "time" on the parametrization of the curve in 2D determines the point in the 2D plane
            n_points (int): points to be created for the cicle in 3D

        Returns:
            Tuple[npt.NDArray, npt.NDArray, float]: points on the circle, normal vecotors of these points, radius of circle
        """
        
        radius, radius_points = self.radius(t)

        # angle between x and y coordinate
        direction = radius_points[2] - radius_points[0]
        if not direction[0] == 0:
            alpha = np.arctan(direction[1] / direction[0])
        else:
            alpha = np.pi if direction[1] >= 0 else -np.pi
        
        # create rotation matirx to rotate points around z
        rot = np.eye(3)
        rot[0,1] = - np.sin(alpha)
        rot[1,1] = np.cos(alpha)
        rot[0,0] = np.cos(alpha)
        rot[1,0] = np.sin(alpha)
        
        # create n_points points
        d_beta = 2 * np.pi / n_points
        points = np.zeros((n_points, 3))
        beta = np.arange(n_points) * d_beta
        points[:,0] = radius * np.cos(beta)
        points[:,2] = radius * np.sin(beta)

        # transform points
        points = np.matmul(points, rot.T)
        
        # for centered data the normals match the vectors
        point_normals = points

        # move circle center
        center = np.zeros(3)
        center[:2] = radius_points[1] # we have the circle in z direction
        points = points + center
        
        return points, point_normals, radius
            
    def cloud_points_normals(self, num_circle_points : int, num_skeleton_points : int) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Compute 3D point Cloud to visualize approximatively 
        how volume is estimated.
        
        Args:
            num_circle_points (int): number of points per circle.
            num_skeleton_points (int): number of circles along skeleton.

        Returns:
            Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: points, normals, radii
        """
       
       
        circle_points = []
        circle_normals = []
        radii = []
       
        for t in np.linspace(0,1,num_skeleton_points):
            points, normals, radius = self.circle(t, num_circle_points)
            circle_points.append(points)
            circle_normals.append(normals)
            radii.append(radius)
        
        points = np.concatenate(circle_points, axis = 0) 
        normals = np.concatenate(circle_normals, axis = 0) 
        radii = np.array(radii, dtype=np.float32)
        
        return points, normals, radii

    @print_time("exporting 3D representation")
    def store_ply(self, path : Path) -> None:
        """Store ply file of the spike at give path
        
        Parent directories will be created if necessary

        Args:
            path (Path): path variable to store mesh
        """
        
        n_points = 20000
        len_rad_ratio = 4.5

        # distribute points accordint to ratio
        num_circle_points = int(np.sqrt(n_points / len_rad_ratio))
        num_skeleton_points = int(len_rad_ratio * num_circle_points)

        # get points and normals, radii
        points, normals, _ = self.cloud_points_normals(num_circle_points, num_skeleton_points)
        
        # determine size of the individual
        lenght = self.lenght(scale=1)
        factors = np.array([2])
        radii = factors * (lenght / num_skeleton_points)
        
        # compute mesh
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
        
        # give some radii of spheres to be fit
        # use some regularity of the mesh and the radii defied above and hope for a good fit
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
               point_cloud, o3d.utility.DoubleVector(radii))

        # store file
        path.parent.mkdir(parents=True, exist_ok=True) 
        o3d.io.write_triangle_mesh(str(path), mesh)