#!/usr/bin/env python3
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import numpy.typing as npt
from typing import Tuple
import click
from icecream import ic
from prep_lib.spike_segmenter import SpikeSegmenter
from prep_lib.abstract_spike import AbstractSpike
import pandas as pd

def compute_spike_integrals(
                dir : Path,
                save_mappings_at : Path,
                save_ply_at : Path,
                save_mask_at : Path,
                scale : float, # one pixel in mm
                segmentation_model : Tuple[str, Path] = ("vit_h", Path("data/models/segment_anything/sam_vit_h_4b8939.pth")),
                segment_only : bool = False,
                abstract_spike_dir : Path | None = None,
                verbose = True,
                ):
    
    """Function to compute abstract representations of each spike in the input folder.
    
    Stores as outputs:
    - volume file
    - 3D modelled ply files
    - segmentation of the spline
    - mask for the segmentation of the spline

    Args:
        dir (Path): input directory
        save_mappings_at (Path): volume mapping file output location.
        save_ply_at (Path, optional): ply output folder location.
        save_mask_at (Path, optional): mask output location.
        scale (float, optional): realworld / image ratio. Defaults to 0.05.
        segment_only (bool, optional): Skip all possible computation if sementation mask exists.
        abstract_spike_dir (Path | None, optional): Store additional plots of abstract spike in
            this directory if given. Defaults to None.
        verbose (bool, optional): print runtimes and progress. Defaults to True.
    """

    # segmenter
    segmenter = SpikeSegmenter(segmentation_model=segmentation_model, verbose=verbose)

    #set mapping path
    mapping_path = save_mappings_at

    #if mapping file already exist, continue with it, otherwhise create an empty one
    if mapping_path.exists():
        df_mapping = pd.read_csv(mapping_path, index_col=0)
    else:
        df_mapping = pd.DataFrame(columns=["img_id", "volume"])
    
    #for n, img in enumerate(Path.iterdir(dir)):
    for n, img in enumerate(dir.iterdir()):

        if img.suffix.lower() != ".jpg":     # <-- only process JPG images
            continue
        if verbose:
            print(f"Process image #{n} {img.name}.")

        #if img.suffix.lower() != ".jpg":     # only process JPG images
        #    continue

        # compute output paths
        ply_path = save_ply_at / f'{img.stem}.ply'
        mask_path_compressed = save_mask_at / f'{img.stem}.npz'
        masked_img_path = save_mask_at / f'{img.name}'


    
        # segment anything is expecting rgb images
        image = cv2.imread(str(img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            if mask_path_compressed.exists():

                if verbose:
                    print(f'Found mask')

                # load existent mask
                #contour_mask = np.load(mask_path_compressed)
                contour_mask = np.load(mask_path_compressed)["arr_0"]

                if not segment_only:
                    # compute abstract spike
                    spike : AbstractSpike = segmenter.initialize_abstract_spline(np_mask_contour=contour_mask)
            else:
                if verbose:
                    print(f'No mask found, have to compute ...')
                if segment_only:
                    # only compute segmentation
                    contour_mask : npt.NDArray = segmenter.segment_spike(rgb_image=image, baseline = True)
                else:
                    # compute segmentation and abstract spike
                    spike : AbstractSpike = segmenter.segment_and_initialize_spike(rgb_image=image, save_at = save_ply_at)
                    contour_mask = spike.contour_mask


            # save mask and contour image
            tmp = deepcopy(image)
            tmp[~(contour_mask.astype(bool))] = 255
            np.savez_compressed(mask_path_compressed, contour_mask.astype(bool))
            plt.imsave(masked_img_path, tmp)


            # Documentation of the method
            if not abstract_spike_dir is None:
                segmenter.save_axis(tmp, Path("data/trees"), img.stem)
            
            if not segment_only:
                # store ply
                spike.store_ply(ply_path)
                # map_volume
                volume = spike.volume(scale=scale)

                
                print(f'Estimated spike volume {volume} mm^3')
                print(volume)
                print(img.stem)

                #add calculated volume to mapping file
                new_entry = pd.DataFrame([{"img_id": img.stem, "volume": volume}])
                df_mapping = pd.concat([df_mapping, new_entry], ignore_index=True)

                if not segment_only: 
                    #df_mapping = df_mapping.drop_duplicates(subset=["img_id"], keep="last")
                    df_mapping.to_csv(mapping_path)


        except Exception as e:
            path = Path('data/baseline_errors.txt')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as stream:
                stream.write(f'{img.name} error: {e} \n')
                stream.close()     


@click.command()
@click.option("--input", type=str, required=True, help="Path to the input images of the spikes, (Note for the script to work there must not be any masks. Bars need to be removed from images!).")
@click.option("--ply-output", type=str, required=True, help="Path to the output masks.(Suggestion: data/lab_images_no_bar)")
@click.option("--mask-output",  type=str, required=True, help="Path to the output images. (Suggestion: data/bar_masks)")
@click.option("--volumes-output",  type=str, required=True, help="Path to the volumes csv file. (Suggestion: data/vol_baseline.csv)")
@click.option("--scale",  type=float, default=0.0496, help="(realworld in mm) / (image in pixel) ratio. Defaults to 0.0496")
@click.option("--segment-only", type=bool, is_flag=True, help="If true, no abstract spikes are created and segmentation is only done for file where no mask is found.")
@click.option("--abstract-spike-dir", type=str, default=None, help="Path: If given, creates plots of abstract spikes in given directory.")
@click.option("--no-verbose",  type=bool, default=False, is_flag=True, help="Print no times and process status.")
def main(input : str,
        ply_output : str,
        mask_output : str,
        volumes_output : str,
        scale : float,
        segment_only : bool,
        abstract_spike_dir : str,
        no_verbose : bool):
    dir = Path(input)
    mask_output = Path(mask_output)
    mask_output.mkdir(parents=True, exist_ok=True)
    ply_output = Path(ply_output)
    ply_output.mkdir(parents=True, exist_ok=True)
    volumes_output = Path(volumes_output)
    volumes_output.parent.mkdir(parents=True, exist_ok=True)
    if not abstract_spike_dir is None:
        abstract_spike_dir = Path(abstract_spike_dir)
        abstract_spike_dir.mkdir(parents=True, exist_ok=True)

    compute_spike_integrals(dir=dir,
                            save_mappings_at=volumes_output,
                            save_ply_at=ply_output,
                            save_mask_at=mask_output,
                            segment_only=segment_only,
                            scale=scale,
                            abstract_spike_dir=abstract_spike_dir,
                            verbose=~no_verbose)

if __name__ == "__main__":
    main()