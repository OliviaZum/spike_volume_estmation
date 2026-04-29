#!/usr/bin/env python3
import matplotlib.pyplot as plt
from prep_lib.bar_remover import BarRemover
from copy import deepcopy
import numpy as np
from pathlib import Path
from typing import Tuple
from icecream import ic
import cv2
import click

def transform_all_images(dir : Path,
                         save_at : Path = Path("data/spikelets_no_bar"),
                         save_mask_at : Path = Path("data/spikelets_no_bar"),
                         skip_exist : bool = False,
                         deepfill_weights_path : Path = Path('data/models/deep_fill/states_pt_places2.pth'),
                         sam : Tuple[str, Path] = ("vit_h", Path("data/models/segment_anything/sam_vit_h_4b8939.pth")),
                         inpaint : bool = True):
    
    """Removes Bars from all images in dir and saves several output files  including bar mask and final image
    
    If some save paths are not given, corresponding output is skipped (althoung it might still get computed!)

    Args:
        dir (Path): Directory to look for files
        save_at (Path, optional): Path to store wheat spikes after inpainting. Defaults to None.
        save_mask_at (Path, optional): Path to store masks. Defaults to None.
        skip_exist (bool, optional): If set true, then computations for files which already exist are skiped
        deepfill_weights_path (Path, optional): Path to the model weights for inpainting. Defaults to Path('data/models/deep_fill/states_pt_places2.pth').
        sam (Tuple[str, Path], optional): Segementation model setup. Defaults to ("vit_h", Path("data/models/segment_anything/sam_vit_h_4b8939.pth")).
        inpaint (bool, optional): Whether or not inpaitnging should be done. Defaults to True.
    """

    save_at.mkdir(parents=True, exist_ok=True)
    inpaint_input = inpaint

    model = BarRemover(segmentation_model=sam,deepfill_weights_path=deepfill_weights_path, verbose=True)
    n = 0

    for img in Path.iterdir(dir):
        print("image: ", end=' ')
        print(img)

        inpaint = inpaint_input
        if not (img.suffix in ['.jpg', '.png', '.jpeg']):
            continue

        n += 1
        image = cv2.imread(str(img))

        # Set paths
        img_name = img.name
        img_npy = f'{img.stem}.npz'
        out_path = save_at / img_name
        out_mask_path = save_mask_at / img_name
        out_mask_path_npy = save_mask_at / img_npy

        if skip_exist and out_mask_path.exists() and out_path.exists() and out_mask_path.exists():
            print(f'Skipping #{n} {img.name} completely')
            continue
        
        if skip_exist and out_mask_path_npy.is_file() and out_mask_path_npy.exists():
            bars_mask = np.load(out_mask_path_npy)
        else:
            bars_mask = None
        
        if skip_exist and  out_path.is_file() and out_path.exists():
            # change inpaint other wise pick input inpaint
            inpaint = False

        
        # Specify the file path
        file_path = Path('data/models/deep_fill/states_pt_places2.pth')

        # Check if the file exists
        if file_path.exists():
            print("File exists")
        else:
            print("File does not exist")
        
        # segment anything is expecting rgb images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        model.remove_bars(rgb_image=image, bar_mask=bars_mask, inpaint=inpaint)      

        image_transformed = model.transformed_image
        bars_mask = model.bars_mask
        
        if not image_transformed is None and not save_at is None:
            plt.imsave(arr = image_transformed, fname=out_path)

        if not bars_mask is None and not (skip_exist and out_mask_path.exists() and out_mask_path_npy.exists()) :
            print(f'Save mask #{n} as {out_mask_path}')
            tmp_img = deepcopy(image)
            
            # color bar mask white
            tmp_img[bars_mask.astype(bool)] = 255
            plt.imsave(arr = tmp_img, fname=out_mask_path)
            np.savez_compressed(out_mask_path_npy, bars_mask.astype(bool))


@click.command()
@click.option("--input", type=str, required=True, help="Path to the input images of which bars must be removed.")
@click.option("--img-output", type=str, required=True, help="Path to the output masks.(Suggestion: data/lab_images_no_bar)")
@click.option("--mask-output",  type=str, required=True, help="Path to the output images. (Suggestion: data/bar_masks)")
@click.option("--no-inpaint", type=bool, is_flag=True , help="If set, no output images will be produced only masks")
@click.option("--skip-exist", type=bool, is_flag=True , help="If set, mask and inpainints are only produced if files do not yet exist")
def main(input : str, img_output : str, mask_output : str, no_inpaint : bool, skip_exist : bool):
    dir = Path(input)
    img_dir = Path(img_output)
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(mask_output)
    mask_dir.mkdir(parents=True, exist_ok=True)
    transform_all_images(dir=dir, save_at=img_dir, save_mask_at=mask_dir, inpaint=(not no_inpaint), skip_exist=skip_exist)
    
if __name__ == "__main__":
    main()



