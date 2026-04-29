from vtk import vtkPLYReader, vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkJPEGWriter, vtkWindowToImageFilter, vtkPolyDataMapper, vtkActor
import math
import click
from pathlib import Path
import numpy as np
from typing import Tuple
import pandas as pd
import re
import shutil
from icecream import ic
import shutil
import glob

IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg"]

def render_ply(file_path : Path, 
               output_folder : Path,
               vertical_views : int,
               horizontal_views : int,
               naming_postfix : str | None = None,
               angle : int = 39,
               file_name_base : str = None,
               output_size : Tuple[int,int] = (3024,4032),
               ):
    
    """Generate images from ply files

    Args:
        file_path (Path): Path to ply file
        output_folder (Path): Path to output folder where to store the files
        vertical_views (int): number of vertical views
        horizontal_views (int): number of horizontal views
        naming_postfix (str) : In case a letter should be added to the image name
        angle (int, optional): angle in degree of camera. Defaults to 41.
        file_name_base (str, optional): File base name used to name files. Defaults to None.
        output_size (Tuple, optional): Output image size (width, height). Defaults to (3024,4032).
    """
    
    if file_name_base is None:
        file_name_base = f'{file_path.stem}_'
    
    # Read the PLY file
    reader = vtkPLYReader()
    reader.SetFileName(file_path)

    # Create a mapper and set its input
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    # Create an actor and set its mapper
    actor = vtkActor()
    actor.SetMapper(mapper)

    # Create a renderer and add the actor
    renderer = vtkRenderer()
    white = (1,1,1)
    blue = (0.0,117/255,244/255)
    b_color = blue
    renderer.SetBackground(b_color[0], b_color[1], b_color[2])  # Background color white
    renderer.AddActor(actor)
    renderer.ResetCamera()

    # Create a render window
    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetOffScreenRendering(1)  # Enable offscreen rendering
    render_window.SetSize(output_size[0],output_size[1])

    # Get the center of bounding box around spike
    center = renderer.GetActiveCamera().GetFocalPoint()
    center = np.array(center)
    
    # Define the radius for the camera orbit
    radius = 200 # corresponds to 200 mm
    
    # define angles
    assert horizontal_views >= 1, "Only positive number of views is allowed"
    assert vertical_views >= 1, "Only positive number of views is allowed"
    phis = np.linspace(start = 0,stop=2 * math.pi, num=horizontal_views, endpoint=False)
    thetas = np.linspace(start = 0, stop=math.pi / 2, num=vertical_views, endpoint=(vertical_views > 1))
    
    if naming_postfix is None:
        naming_postfix = ""
    else:
        naming_postfix = f"_{naming_postfix}"
    
    # horizontally camera start is at (radius, 0, 0)
    for horizontal_idx, phi in enumerate(phis):
        for vertical_idx, theta in enumerate(thetas):
            print("View " + str(vertical_idx) + "_" + str(horizontal_idx))
            if vertical_views > 1:
                output_path = output_folder / f'{file_name_base}{horizontal_idx}_{vertical_idx}{naming_postfix}.jpg'
            else:
                output_path = output_folder / f'{file_name_base}{horizontal_idx}{naming_postfix}.jpg'
            
            # Calculate spherical coordinates
            camera_x = center[0] + radius * math.cos(theta) * math.cos(phi)
            camera_y = center[1] - radius * math.sin(theta)
            camera_z = center[2] + radius * math.cos(theta) * math.sin(phi)

            camera = renderer.GetActiveCamera()
            near_clip = 1  # Near clipping plane (should be > 0)
            far_clip = 10000  # Far clipping plane (should be > near_clip)
            camera.SetClippingRange(near_clip, far_clip)

            camera = renderer.GetActiveCamera()
            camera.SetClippingRange(near_clip, far_clip)
            camera.SetPosition(camera_x, camera_y, camera_z)
            camera.SetFocalPoint(center)
            camera.SetViewUp(0, -1, 0)  # Z-axis is up
            camera.SetViewAngle(angle) # heuristically with images

            # Render the scene
            render_window.Render()

            # Capture and save the image
            w2if = vtkWindowToImageFilter()
            w2if.SetInput(render_window)
            w2if.Update()

            writer = vtkJPEGWriter()

            if vertical_views > 1:
                output_path = output_folder / f'{file_name_base}{horizontal_idx}_{vertical_idx}{naming_postfix}.jpg'
            else:
                output_path = output_folder / f'{file_name_base}{horizontal_idx}{naming_postfix}.jpg'
            
            writer.SetFileName(str(output_path))
            writer.SetInputConnection(w2if.GetOutputPort())
            writer.Write()


def create_mapping_artificial(volume_path : Path, image_path : Path):

    """Create mapping given a folder with artifically created
    images and a mapping csv file.
    
    The `volume_path` needs to contain a .txt file for each plant of which
    images are found in `image_path`. Images for which no .txt is found will
    not appear in the output file but will generate a error_message in the
    `error.txt` placed the directory `image_path`.
    
    `error.txt` will be deleted in the `image_path` in every run of the script.
    
    In the image_path the images need to adhere the naming convention for
    artifical images of plant x: 
    
    `image_name = 'x[_p][_t][_a].jpg'`
    
    Creates a `vol_mapping.csv` file in the image_path directory.
    
    where 
    x is the plant_id,
    p is a number indicating the angle on the x,y plane (or the index of the angle), 
    t is the angle between (x,y)-plane and the camera position (or the index hereof),
    a is some desciption of the image,
    Note that only x is important for this function, all the other naming convention parameters
    are optional.

    Args:
        volume_path (Path): Path to directory containing the .txt file indicating volumes of plants
        image_path (Path): Path to the folder containing the images.
    """
    
    error_file_path = image_path / "error.txt"
    if error_file_path.exists(): # delete error file if existent
        error_file_path.unlink()

    if not image_path.exists():
        raise FileExistsError(f"Image path {str(image_path)} does not exist!\n")

    if not volume_path.exists():
        raise FileExistsError(f"Volume path {str(volume_path)} does not exist!\n")
    
    volume_list = []
    image_name_list = []
    for i, file in enumerate(image_path.iterdir()):
        
        if file.suffix in IMAGE_EXTENSIONS:
            # image found
            
            match = re.search(r'^\d{1,2}_\d{1,2}_\d{1,2}', file.name)
            if match is None:
                # write error
                with error_file_path.open("a") as stream:
                    stream.write(f"Image #{i}: {file.name} does not adhere naming convention.\n")
                    stream.close()
                # next file
                continue
        
            current_plant_id = match[0]
            
            # search for volume in volume_path
            current_plant_volume_path = volume_path / f'{current_plant_id}.txt'
            if not current_plant_volume_path.exists():
                # write error
                with error_file_path.open("a") as stream:
                    stream.write(f"Volume file for Image #{i}: {file.name} not found in {str(volume_path)}.\n")
                    stream.close()

            try:
                current_volume = float(current_plant_volume_path.read_text())
            except:
                # write error
                with error_file_path.open("a") as stream:
                    stream.write(f"Volume file for Image #{i}: {file.name} could not be read.\n")
                    stream.close()
                continue
            
            image_name_list.append(file.name)
            volume_list.append(current_volume)
    
    csv_dict = {
        "img_name": image_name_list,
        "volume" : volume_list
        }
    
    output_path = image_path / "vol_mapping.csv"
    mapping_df = pd.DataFrame.from_dict(csv_dict) 
    mapping_df.to_csv(output_path, index=False)


def merge_datasets_(dataset_path_one : Path, dataset_path_two : Path, output_path : Path):
    """Merges two datasets
    
    Both datasets need to contain images and a mapping file named `vol_mapping.csv`.
    
    Output directory will be created, error is thrown if it already exists.

    Args:
        dataset_path_one (Path): First dataset to be merged
        dataset_path_two (Path): Second datset to be merged
        output_path (Path): Output where to store the merged dataset
    """
    
    if not dataset_path_one.exists():
        raise FileExistsError(f"Dataset {str(dataset_path_one)} does not exist!")
    
    if not dataset_path_two.exists():
        raise FileExistsError(f"Dataset {str(dataset_path_two)} does not exist!")

    if output_path.exists():
        raise FileExistsError(f"Output dataset {str(output_path)} already exist!")
    output_path.mkdir(parents=True)
    
    mapping_one = dataset_path_one / "vol_mapping.csv"
    mapping_two = dataset_path_two / "vol_mapping.csv"
    
    if not mapping_one.exists():
        raise FileExistsError(f"No mapping {str(mapping_one)} found.")

    if not mapping_two.exists():
        raise FileExistsError(f"No mapping {str(mapping_two)} found.")
    
    df_mapping_one = pd.read_csv(mapping_one)
    df_mapping_two = pd.read_csv(mapping_two)
    
    df_merged = pd.concat([df_mapping_one, df_mapping_two], axis=0, ignore_index=True)
    if not 'img_name' in df_merged.columns or not 'volume' in df_merged.columns:
        raise FileExistsError(f"Merge failed. Check vol_mapping files for columns img_name and volume.")
    df_merged.drop_duplicates(subset=["img_name"])
    
    print(f"Start copying files from {dataset_path_one.name}")
    for file in dataset_path_one.iterdir():
        if file.suffix in IMAGE_EXTENSIONS:
            dst = output_path / file.name
            shutil.copy(file, dst)
    
    print(f"Start copying files from {dataset_path_two.name}")
    for file in dataset_path_two.iterdir():
        if file.suffix in IMAGE_EXTENSIONS:
            dst = output_path / file.name
            shutil.copy(file, dst)
            
    df_merged.to_csv(output_path / "vol_mapping.csv", index=False)
            

@click.command()
@click.option("--data", "-d", required=True, type=str, help="Path to the input data folder containing the .ply files")
@click.option("--output", "-o", type=str, required=True, help="Path to the output folder. (Will be created if not existant.)")
@click.option("--views", "-v", type=int, required=True, help="Number of views of each .ply")
@click.option("--naming-postfix", type=str, required=True, help="Creates names like xx_xx_xx_xx[_xx]_<postfix>.jpg")
@click.option("--spherical", "-s", type=bool, is_flag=True, default=False, help="If set, camera position also changes in z direction, otherwise only circulare around the object.")
@click.option("--outputsize", "-os", type=(int,int), default=(3024,4032), help="Size of the images")
@click.option("--angle", type=float, default=35, help="Camera angle (mostly tunig parameter depends on camera intrinsicts which are often not known.)")
@click.option("--volumes-dir", type=str, default=None, help="Path to directory with volumes in txt file for each spike")
def main(data : str, 
         output : str, 
         views : int, 
         naming_postfix : str, 
         spherical : bool, 
         outputsize : Tuple[int,int],
         angle : int,
         volumes_dir : str | None) -> None:
    
    """Creates images from 3D .ply files. Default is to move the camera
    circularly around the object and as an option, one can use a spherical
    camera positions also changing the z position of the camera.
    outputsize: 256, 256.
    """

    data_path = Path(data)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True) # create if it does not yet exists
    
    if spherical:
        vertical_views = views    
        horizontal_views = views    
    else:
        vertical_views = 1    
        horizontal_views = views    

    for stl_file in data_path.iterdir():
        if stl_file.suffix == '.ply':
           
            
            print("Generating views for file " + stl_file.name)
            render_ply(stl_file,
                       output_path,
                       vertical_views=vertical_views,
                       horizontal_views=horizontal_views,
                       output_size=outputsize,
                       angle=angle,
                       naming_postfix=naming_postfix)
    
    if not volumes_dir is None:
        volumes_dir = Path(volumes_dir)
        create_mapping_artificial(volume_path=volumes_dir, image_path=output_path)
        
@click.command()
@click.option("--dataset", required=True, type=str, help="Path to dataset (must contain images and vol_mapping.csv file)")
@click.option("--volumes-dir", type=str, default=None, help="Path to directory with volumes in txt file for each spike")
def generate_dataset_volumes(dataset : str, volumes_dir : str):
    dataset = Path(dataset)
    volumes_dir = Path(volumes_dir)
    create_mapping_artificial(image_path=dataset, volume_path=volumes_dir)
        
@click.command()
@click.option("--dataset-1", required=True, type=str, help="Path to first dataset (must contain images)")
@click.option("--dataset-2", required=True, type=str, help="Path to second dataset (must contain images and vol_mapping.csv file)")
@click.option("--output-dataset", required=True, type=str, help="Path to output directory of new dataset must not exist yet.")
def merge_datasets(dataset_1 : str, dataset_2 : str, output_dataset : str):
    dataset_1 = Path(dataset_1)
    dataset_2 = Path(dataset_2)
    output_dataset = Path(output_dataset)
    merge_datasets_(dataset_1, dataset_2, output_dataset)
    

@click.command()
@click.option("--data", required=True, type=str, help="Path to the input data folder containing the .ply files")
@click.option("--output", type=str, required=True, help="Path to the output folder. (Will be created if not existant.)")
def open_lrm(data : str, output : str):

    """Creates training data for open lrm model
    
    For each ply a folder with 360 images is created, one for each degree. The folder will be placed
    in the output_base directory. The output_base directory itself is created if it does not yet exist.

    Args:
        data (str): data input directory
        output (str): image output directory
    """

    data_path = Path(data)
    output_base = Path(output)
    output_base.mkdir(parents=True, exist_ok=True) # create if it does not yet exists
    
    vertical_views = 1    
    horizontal_views = 20    

    for stl_file in data_path.iterdir():
        if stl_file.suffix == '.ply':
            print("Generating views for file " + stl_file.name)
            output_file_base = output_base / stl_file.stem
            output_file_base.mkdir(parents=True, exist_ok=True)
            render_ply(stl_file,
                       output_folder=output_file_base,
                       vertical_views=vertical_views,
                       horizontal_views=horizontal_views,
                       angle=30,
                       file_name_base='',
                       output_size=(256,256))

if __name__ == "__main__":
    main()