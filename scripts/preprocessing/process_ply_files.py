
import open3d as o3d
import numpy as np
import pymeshlab
from pathlib import Path
import click
import numpy.typing as npt
from icecream import ic
from typing import Tuple
import numpy.typing as npt
from lib.function_timer import print_time
import shutil
import re
import pandas as pd

#plotting functions: 
def visualize_pyramid_base(mesh, base_y, save_path):

    vertices = np.asarray(mesh.vertices)

    # ---- select band around base_y
    y_base_range = 10
    y_mask = (vertices[:,1] >= base_y - y_base_range/2) & (vertices[:,1] <= base_y + y_base_range/2)

    base_vertices = vertices[y_mask]

    if len(base_vertices) == 0:
        print("No base points found.")
        return

    # ---- compute center
    base_center_x = (np.max(base_vertices[:,0]) + np.min(base_vertices[:,0])) / 2
    base_center_z = (np.max(base_vertices[:,2]) + np.min(base_vertices[:,2])) / 2
    base_center_y = base_y

    # ---- base slice point cloud
    base_pcd = o3d.geometry.PointCloud()
    base_pcd.points = o3d.utility.Vector3dVector(base_vertices)

    # ---- center sphere
    center = o3d.geometry.TriangleMesh.create_sphere(radius=2)
    center.translate([base_center_x, base_center_y, base_center_z])

    # ---- renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(1024,1024)
    scene = renderer.scene

    # ---- mesh material (transparent)
    mesh_mat = o3d.visualization.rendering.MaterialRecord()
    mesh_mat.shader = "defaultLitTransparency"
    mesh_mat.base_color = [0.7,0.7,0.7,0.3]

    scene.add_geometry("mesh", mesh, mesh_mat)

    # ---- base points material
    point_mat = o3d.visualization.rendering.MaterialRecord()
    point_mat.shader = "defaultUnlit"
    point_mat.point_size = 2

    base_pcd.paint_uniform_color([1,0,0])
    scene.add_geometry("base", base_pcd, point_mat)

    # ---- center sphere material
    center_mat = o3d.visualization.rendering.MaterialRecord()
    center_mat.shader = "defaultLit"
    center.paint_uniform_color([0,1,0])

    scene.add_geometry("center", center, center_mat)

    bbox = mesh.get_axis_aligned_bounding_box()
    scene.camera.look_at(
        bbox.get_center(),
        bbox.get_center() + [0,0,200],
        [0,1,0]
    )

    # ---- save perspective view
    img = renderer.render_to_image()
    o3d.io.write_image(str(save_path), img)

    # ---- TOP VIEW
    scene.camera.look_at(
        [base_center_x, base_center_y, base_center_z],
        [base_center_x, base_center_y + 200, base_center_z],
        [0,0,1]
    )

    img_top = renderer.render_to_image()

    top_path = str(save_path).replace(".png", "_top.png")
    o3d.io.write_image(top_path, img_top)

class BaseNotFoundError(Exception):
    pass

def find_pyramid_base(mesh : o3d.geometry.TriangleMesh, base_y : float) -> Tuple[float, float, float]:

    """Find base of block. This base point is defined as the
    point which has y coordinate on the level with the most denstity
    in this direction. In the x and z direction it then takes the mid value,
    which is the average of the minimal and maximal point in this direction.

    Throws an error if no point in base.

    Args:
        model_path (o3d_mesh): path to the file

    Returns:
        Tuple[float, float, float]: Triple with base point coordinates
    """

    # Load the 3D model
    vertices = np.asarray(mesh.vertices)

    # Find the vertex with the maximum y-coordinate as the base
    base_index_y = base_y

    # Create a boolean mask for vertices within the specified y-axis range
    y_base_range = 10
    y_mask = (vertices[:,1] >= base_index_y - y_base_range / 2) & (vertices[:,1] <= base_index_y + y_base_range / 2)

    # Apply the mask to filter vertices
    base_vertices = vertices[y_mask]
    if not base_vertices.ndim == 2:
        raise BaseNotFoundError("Base not found.")

    # Calculate the mean of x and y coordinates to find the center
    base_center_x = (np.max(base_vertices[:,0]) + np.min(base_vertices[:, 0])) / 2
    base_center_z = (np.max(base_vertices[:,2]) + np.min(base_vertices[:, 2])) / 2
    base_center_y = base_index_y

    return base_center_x, base_center_y, base_center_z

def create_cone(base_center : npt.NDArray, cone_radius : float, cone_height : float, cone_cap_height : float) -> npt.NDArray:
    """Compute point cloud of cone, mainly to debug

    Args:
        base_center (npt.NDArray): Base point of the cone
        cone_radius (float): Cone radius
        cone_height (float): Cone height
        cone_cap_height (float): Cone height where to cap the apex

    Returns:
        npt.NDArray: point cloud
    """

    # note that vertical will be in y direction
    n_vertical_stages = 20
    n_circle_points = 40
    
    points = []
    for h in np.linspace(0,cone_height, n_vertical_stages):
        # the radius is a linear funciton of the height
        # r(0) = 0, r(cone_height) = cone_radius
        # r(h) = h / cone_height * cone_radius
        radius_h = h / cone_height * cone_radius
       
        alphas =  np.linspace(0, 2 * np.pi, n_circle_points)
        x = radius_h * np.cos(alphas) 
        y = np.maximum(h - cone_height, cone_cap_height - cone_height) * np.ones_like(x)
        z = radius_h * np.sin(alphas)
        points.append(np.stack([x,y,z],axis=1))
    
    points = np.concatenate(points, axis=0)
    
    zero_point = np.zeros(3).reshape(1,3)
    points = np.concatenate([points, zero_point], axis=0)
    
    np_points = np.array(points)
    np_points_in_space = base_center + np_points

    return np_points_in_space


@print_time("remove cone")
def remove_cone(mesh : o3d.geometry.TriangleMesh, 
                type : str, 
                base_center : npt.ArrayLike,
                cone_radius : float,
                cone_height : float,
                cone_cap_height : float,
                plot_cone : bool = False,
                verbose : bool = True,
                ) -> Tuple[pymeshlab.MeshSet, float]:
    """Removes cone from mesh and returns

    Args:
        mesh (o3d.geometry.TriangleMesh): open3d mesh containing the points and vertices
        type (str): color information of ply file 
        base_center (npt.ArrayLike): cone center point
        cone_radius (float): cone height
        cone_height (float): cone radius
        cone_cap_height (float): distance from apex in y direction to cut off cone
        cone_base_y (float): expected base of cone (manually entered for more stability)
        plot_cone (bool, optional): indicator if cone should be plotted
        verbose (bool, optional): whether to print runtime . Defaults to True

    Returns:
        Tuple[o3d.geometry.TriangleMesh, float]: open 3d mesh with cutted spike and computed mesh volume
    """
    
    if plot_cone:
        # compute and plot cone for debugging
        cone_points = create_cone(base_center=base_center,
                                  cone_radius=cone_radius,
                                  cone_height=cone_height,
                                  cone_cap_height=cone_cap_height)
        cone_cloud = o3d.geometry.PointCloud()
        cone_cloud.points = o3d.utility.Vector3dVector(cone_points)
        colors = np.array([[230,0,0]]*cone_points.shape[0], dtype=np.float64)
        cone_cloud.colors = o3d.utility.Vector3dVector(colors)
        mesh.compute_vertex_normals()
        # Save the new mesh to a file
        o3d.visualization.draw_geometries([mesh, cone_cloud])

    # Load the 3D model
    vertices = np.asarray(mesh.vertices)

    #save color information of ply file if there is any
    if type == 'color':
        colors = np.asarray(mesh.vertex_colors)
        # Combine vertices and colors horizontally to form a single array
        combined_data = np.hstack((vertices, colors))
        # Create a DataFrame from the combined array
        columns = ['X', 'Y', 'Z', 'R', 'G', 'B']
        df = pd.DataFrame(combined_data, columns=columns)
        ic(df)

    # Define the region to remove (vertices within a circular base and below the apex)
    
    # take base center (note that x0 = x1, z0 = z1 where p0 = (x0,y0,z0)= apex and p1 = (x1,y1,z1) = base_center
    x, y1, z = base_center # defines p1 = base_center
    
    # get p1 = apex i.e only y1 needs computation (note that this must be apex of surrounding cone)
    y0 = y1 - cone_height # defines p0 apex
    
    # base on fundamental properties of triangles we have r = h * (base_radius / cone_height) and (base_radius / cone_height) = tan_alpha
    tan_a = cone_radius / cone_height
    
    # we only want to remove points "above" apex
    mask_above = vertices[:,1] >= y0 + cone_cap_height
    points_above = vertices[mask_above]
    
    # take height of every point with respect to apex ([:,1] takes y column)
    h = points_above[:, 1] - y0
    
    # computes distance to center in (x,z) plane (of points we want to test if in real cone or not)
    d_to_center = np.sqrt((points_above[:,0] - x)**2 + (points_above[:,2] - z)**2)
    
    # radius of real cone at the heights of the points we want to test for
    cone_rad_at_h = h * tan_a
    
    # if point radius is smaller than real cone radius mask as point in cone
    mask_in_cone = cone_rad_at_h >= d_to_center
    
    # make mask larger to mach input size
    mask = mask_above
    mask[mask_above] = mask_in_cone

    # Invert the mask to keep vertices outside the cone
    mask_invert = ~mask

    # Create a mapping from old vertex indices to new indices
    vertex_mapping = np.zeros(len(vertices), dtype=int)
    vertex_mapping[mask_invert] = np.arange(np.sum(mask_invert))  # Indices for vertices to keep

    # Filter vertices, triangles and color information based on the mask
    vertices_to_keep = vertices[mask_invert]
    mesh_triangles = np.array(mesh.triangles)
    triangle_mask = np.all(mask_invert[mesh_triangles], axis=1)
    triangles_to_keep = mesh_triangles[triangle_mask]
    triangles_to_keep = vertex_mapping[triangles_to_keep]

    # Check if any vertices are referenced by triangle indices outside the valid range
    if np.max(triangles_to_keep) >= len(vertices_to_keep):
        print("Error: Triangle indices reference vertices outside the valid range.")
        return
    

    # create pymeshlab mesh
    np_vertices = np.array(vertices_to_keep, dtype=np.float64)
    np_faces = np.array(triangles_to_keep, dtype=np.float64)
    pm_mesh = pymeshlab.Mesh(vertex_matrix=np_vertices, face_matrix=np_faces)
    pm_meshset = pymeshlab.MeshSet()
    pm_meshset.add_mesh(pm_mesh)

    # remove artefacts
    pm_meshset.generate_splitting_by_connected_components(delete_source_mesh=True)
    face_numbers = []
    for m in pm_meshset:
        face_numbers.append(m.face_number())
    max_faces = np.max(face_numbers)

    pm_meshset.meshing_remove_connected_component_by_face_number(mincomponentsize=max_faces)
    

    # remove empty meshes
    while pm_meshset.current_mesh().face_number() == 0:
        pm_meshset.delete_current_mesh()
    

    # fill holes (smoothing helps to fix irregular cases)
    #pm_meshset.apply_coord_hc_laplacian_smoothing()
    pm_meshset.meshing_close_holes(maxholesize=300) 
    mesh_measures = pm_meshset.get_geometric_measures()
    
    #if volume can be calculated, save it in mesh_volume variable
    #if volume cannot be calculated, try to smooth it again up to 10 times. 
    #This is not done for color ply files, because it slightly shifts the coordinates of the vertices
    if 'mesh_volume' in mesh_measures:
        mesh_volume = mesh_measures['mesh_volume']
        ic(mesh_volume)
    else:
        mesh_volume = -1
        
        if type == 'no_color':
            for i in range(10):
                ic(i)
                if i <= 8:
                    pm_meshset.apply_coord_hc_laplacian_smoothing()
                    pm_meshset.meshing_close_holes(maxholesize=300)
                    mesh_measures = pm_meshset.get_geometric_measures()
                else:
                    pm_meshset.apply_coord_hc_laplacian_smoothing()
                    pm_meshset.meshing_close_holes(maxholesize=300, refinehole = True)
                    mesh_measures = pm_meshset.get_geometric_measures()
                    
                if 'mesh_volume' in mesh_measures:
                    print("Condition met. Leaving the loop.")
                    mesh_volume = mesh_measures['mesh_volume']
                    break
                else:
                    print("Volume still -1")
                    mesh_volume = -1
        
            
    # store faces, vertices and color
    kept_faces = pm_meshset.current_mesh().face_matrix() 
    kept_vertices = pm_meshset.current_mesh().vertex_matrix()


    #make kept_vertices to a data frame to exract the color information from the original dataframe
    #that we want to re-add
    # Create a DataFrame with X, Y, Z columns
    if type == 'color':
        df_kept_vertices = pd.DataFrame(kept_vertices, columns=['X', 'Y', 'Z'])

        #keep color of necessary vertices:
        # Merge the large and small DataFrames based on vertices (X, Y, Z)
        filtered_df = pd.merge(df, df_kept_vertices, on=['X', 'Y', 'Z'])
    
        # Extract the last three columns ('R', 'G', 'B') and transform them to a NumPy array
        rgb_array = filtered_df[['R', 'G', 'B']].to_numpy()
        ic(rgb_array)
        
    kept_mesh = o3d.geometry.TriangleMesh()
    kept_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(kept_vertices))
    kept_mesh.triangles = o3d.utility.Vector3iVector(kept_faces)
    
    #if type == color, add the color information to the ply file 
    if type == 'color':
        kept_mesh.vertex_colors = o3d.utility.Vector3dVector(rgb_array)
    
    return kept_mesh, mesh_volume

#cone height: 27 in 2024 / 33 in 2023
@click.command()
@click.option("--input-dir", required=True, type=str, help="Input directory containing the ply files.")
@click.option("--output-dir", required=True, type=str, help="Output directory containing the ply files and the volume files.")
@click.option("--cone-radius", default=28, type=float, help="Base radius of cone to be removed. Defautls to 28.")
@click.option("--cone-height", default=33, type=float, help="Height of cone to be removed (apex to base radius). Defaults to 33.")
@click.option("--cone-cap-height", default=7, type=float, help="Height to be cut form the peak of cone. Defautls to 7.")
@click.option("--cone-base-y", default=80, type=float, help="Base y coordinate of cone to be removed. Defaults to 80.")
@click.option("--type", default = "no_color", type=str, help="Color information of spikes, color or no_color. Defaults to no_color.")
@click.option("--plot-cone", is_flag=True, help="If true show each spike with the cone to be removed (mainly used to deterimine cone height and radius).")
@click.option("--no-verbose", required=False, default=False, is_flag=True, help="Do not show computing times and spike to be computed.")
def process_ply_files_in_root_directory(input_dir : Path,
                                        output_dir : Path,
                                        type : str,
                                        cone_radius : float,
                                        cone_height : float,
                                        cone_cap_height : float,
                                        cone_base_y : float,
                                        plot_cone : bool,
                                        no_verbose : bool,
                                        ):

    verbose = ~no_verbose
    output_dir = Path(output_dir)
    input_dir = Path(input_dir)
    n = 0

    # Delete the existing error file if it exists
    error_file_path = output_dir / "error.txt"
    if error_file_path.exists():
        error_file_path.unlink()

    # Process each PLY file
    for file in input_dir.iterdir():
        
        if not file.is_file() or not file.suffix == ".ply":
            # if file is not a file or it is not a ply file, then go to next file
            continue

        n += 1
        if verbose:
            print(f"Process {file.name} #{n}", end=". ")

        output_path = output_dir / file.name
        output_dir.mkdir(parents=False, exist_ok=True)
        output_volume_path = output_dir / f'{file.stem}.txt'

        # Check if the output file already exists. If it already exists, skip this file and 
        # go to next file
        if output_path.exists() and output_volume_path.exists():
            with open(output_volume_path, 'r') as f:
                content = f.read().strip()
                volume = float(content)  # Assuming the content is a numerical value

            if volume > 0:
                if verbose:
                    print("File already exists and volume is larger than 0")
                continue

        def handle_error(e):
            # Write error message to the error file
            error_message = f"Error processing {file.name}: {str(e)}\n"
            with (error_file_path).open(mode="a") as f:
                f.write(error_message)
                f.close

        try:
            mesh = o3d.io.read_triangle_mesh(str(file))
            pyramid_base_center = find_pyramid_base(mesh, base_y=cone_base_y)
            # visualize base detection
            visualize_pyramid_base(
                mesh,
                base_y=cone_base_y,
                save_path=output_dir / f"{file.stem}_base_debug.png"
            )   

        except BaseNotFoundError as e:
            error_message = f"Warning processing {file.name}: Base not found\n"
            with (error_file_path).open(mode="a") as f:
                f.write(error_message)
                f.close
            pyramid_base_center = (0,140,0) # base should not harm anything
        except Exception as e:
            handle_error(e)

        try:
           
            
            out_mesh, out_volume = remove_cone(mesh=mesh, 
                                 type=type,
                                 base_center=pyramid_base_center,
                                 cone_radius=cone_radius,
                                 cone_height=cone_height,
                                 cone_cap_height=cone_cap_height,
                                 plot_cone=plot_cone,
                                 verbose=verbose,
                                 )
        
            # write results
            if out_volume < 0:
                with (error_file_path).open(mode="a") as f:
                    f.write(f"{file.name}\n")
                    f.close()
                    
            out_mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([out_mesh])
            o3d.io.write_triangle_mesh(str(output_path), out_mesh)
            output_volume_path.write_text(f'{out_volume}')

        except Exception as e:
            handle_error(e)


@click.command()
@click.option("--input-dir", required=True, type=str, help="Input directory i.e. root folder of scans")
@click.option("--output-dir", required=True, type=str, help="Output directroy for ply scans `flattened` in one directory without nesting")
def flatten_ply_scans(input_dir : Path | str, output_dir : Path | str):
    
    """Take scans out of their nested structure and move them into
    a new directory. Only the ply files are copied, with the name of the
    subfolder in the original nested folder structure

    The folder sturcture is assumed to be as follows e.g.:
    
    
    input_dir
        10_5_1
            ....jpg
            ....jpg
            Scan.ply
        11_6_2
            ....jpg
            ....obj
            Scan.ply
        .
        .
        .
    
    The above will be converted to
    
    output_dir
        10_5_1.ply
        11_6_2.ply
        .
        .
        .
        

    Args:
        input_dir (Path | str): input `root of scans` directory 
        output_dir (Path | str): output directory of only the ply scans
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # create output direcotry if it does not exists yet
    output_dir.mkdir(parents=True, exist_ok=True)
    cnt = 0
    cnt_exists = 0

    # root directory of scans
    for dir in input_dir.iterdir():
        
        # for each spike `dir` we expect a subdirecotry
        if dir.is_dir():
            for f in dir.iterdir():
                if f.is_file() and f.suffix == '.ply':
                    
                    name = dir.name
                    # Use a regular expression to match the desired pattern
                    new_name = re.search(r'^.*_.*_\d{1,2}', name)[0]
                    #new_name = re.search(r'^\d{1,2}', name)[0]
                    cnt += 1

                    out_path = output_dir / f'{new_name}.ply'
                    if out_path.exists():
                        cnt_exists += 1
                        print(out_path)
                    
                    shutil.copy(str(f), str(output_dir / f'{new_name}.ply')) 

    print(f"Found Files : {cnt}, files existed : {cnt_exists}")


if __name__ == "__main__":
    process_ply_files_in_root_directory()


#spikes 2024: cone_height = 27
#spikes 2023: cone_height = 33