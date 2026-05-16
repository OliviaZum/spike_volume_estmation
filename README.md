# Code folder

We use [Poetry](https://python-poetry.org) to track dependencies.
Add dependencies with `poetry add <package>` to make sure it is
tracked. You can run the code in an environment according to the
`poetry.lock` file with `poetry run <file>`.

# Description of the project

Volume estimation of wheat spikes is an important task to improve yield to mitigate adverse impacts of climate change and the effects of an increasing world population. A dataset comprising about 1700 wheat spikes volumes and corresponding images underwent preprocessing before being utilized for volume estimation. During the preprocessing stage, the cone needs to be removed from the scans in order to extract a precise volume. Furthermore, bars are removed from the images, the spikes are inpainted and segmented in order to increase accuracy in volume prediction and spikelet counting. 

Baseline model: 
We developed a first baseline approach that computes the number of pixels belonging to the spike. We developed a second baseline algorithm that computes cylindrical volumes at small intervalls along the spike. These baseline models were used to answer the question if neural networks perform better than a baseline to estimate the wheat spike volume.      

Neural networks: 
The dateset consists of images and scans of about 1700 wheat spikes. To gain more information, the spikes were imaged from 6 different angles. 

## Installation and setup

Make sure to download and install the necessary files:
- download [deepfill_model](https://drive.google.com/u/0/uc?id=1L63oBNVgz7xSb_3hGbUdkYW1IuRgMkCa&export=download) and move the `.pth` file to the folder `code/data/models/deepfill/`. It should be called `states_pt_places2.pth`.
- download segment anything weights: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth). They have to be placed in `data/models/segment_anything/sam_vit_h_4b8939.pth`.
- download [FoMo4Wheat](https://github.com/PheniX-Lab/FoMo4Wheat) and place it in `data/models/backbone`. It should be names `FoMo4Wheat_base.pth`. Adapt the path in neural_nets.py if you use a different model.

## Scripts

#### gen-dataset
This script can be executed in the poetry environment (entered with `poetry shell`) with e.g. the command:
```
gen-dataset -d data/Scans -o data/Scans_out -v 3 --outputsize 256 256
```
Which will generate 3 views of each spike by placing the camera around the spike in a circular fashion.
Note that with the flag `-s` or `--spherical` the camera can be placed anywhere around the spike.
For more information call `gen-dataset -h`.

#### rm-bars
This script can be executed in the poetry environment (entered with `poetry shell`) with e.g. the command:
```
rm-bars --input data/lab_images --img-output data/lab_images_no_bar --mask-output data/lab_images_mask
```
For each image in the `--input` folder, it will create a mask for the bars and store it in `--mask-output`,
moreover it will generate for each image a new image with the bars removed and inpainted.

For this to work it is important to download pretained weights for segmentation and imputing in the following locations:

Segment anything weights: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
They have to be placed in `data/models/segment_anything/sam_vit_h_4b8939.pth`

Deepfill weights: [Deepfill Model](https://drive.google.com/u/0/uc?id=1L63oBNVgz7xSb_3hGbUdkYW1IuRgMkCa&export=download).
They have to be placed in `data/models/deep_fill/states_pt_places2.pth`

Note that the origninal weights are trained on a tensorflow model. We are using the weights converted by the reimplementation
in pytorch [original reimplementation](https://github.com/nipponjo/deepfillv2-pytorch). As this repostory is not set up for
installation, the actual code we then use in the script is forked form the latter repository and adapted for our use,
[fork of reimplementation](https://github.com/ncograf/deepfillv2-pytorch).

#### cut-spikes
This script can be executed in the poetry environment (entered with `poetry shell`) with e.g. the command:
```
cut-spikes --input-dir data/special_scans --output-dir data/special_scans_no_cone --plot-cone --type no_color
```
Use `cut-spikes --help` see all options.

The `--plot-cone` flag is to plot every single spike with the cone which will be removed.
There are more opions ot determine the cone height and radius.

Note that the script works for any type of scans, no matter, wheter the spike was fixed on the side of the block
or on top of it.

Type color needs to be set to color or no_color, depending on the ply files. If the color information is available, it will be re-added to the ply file if type is set to color


#### gen-base
This file is used to generate the base line volumes using some "cylindrical" fitting of the splines.
Input should be a directory with images (note that for this imput it is assumed that the
bars are already removed) and output will then be a the masked spikes, the ply files of the
3d models according to the baseline estimation and a spike to volume mapping file.

```
gen-base --input data/lab_images_no_bar --ply-output data/lab_image_plys --mask-output data/lab_images_spike_mask --volumes-output data/volume_baseline.csv
```

For this to work it is important to download pretained weights for segmentation and imputing in the following locations:

Segment anything weights: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
They have to be placed in `data/models/segment_anything/sam_vit_h_4b8939.pth`

#### flat-scans
This script is to convert the nested folder struction, which is the output of the original ply
scans to a flatted files structure. I.e in the input directory we assume to have one folder for
each spike and in this folder a ply file containing the mesh of the spike. This is then
simply transformed to a folder containing only the ply files with the corresponding names.

```
flat-scans --input-dir data/scans_11_12_2023 --output-dir data/scans_11_12_2023_flat
```

#### rm-invalid-vol

This script works (just as all other scripts) in the poetry shell.
It removes all mappings form the given mapping input file with volumes == -1.
Such mapping occured because for some spikes the 3D scans were not good enough to extract the volume.

```
rm-invalid-vol --mapping-file data/vol_mapping.csv
```


#### merge-datasets

Merges two datasets from to folders into one big new dataset
Each dataset is a directory with images and a vol_mapping.csv file
```
merge-dataset --dataset-1 data/dataset1 --dataset-2 data/dataset2 --output-dataset data/output_dataset
```

## Estimate Spike Volume 

#### Area Baseline

This script creates an area-based baseline for spike volume prediction. It reads segmentation masks stored as .npz files, counts the number of foreground pixels for each image, and uses the average foreground area across a fixed number of images per plant as a simple prediction feature. It filters out failed segmentations based on very small or very large foreground areas. The script then keeps only plants that are present in the train or test mapping files, saves CSV files with area and volume values, and creates a histogram of the foreground-pixel distribution.

Run the file

```
area-baseline --input '/path/to/dataset/baseline_jpg_ply_npz_without_stalk' --output 'data/output_baseline' --mapping_test 'data/path/to/mapping_test.json' --mapping_train 'data/path/to/mapping_train.json' --num_imgs 4

```

#### Geometric Baseline

This script evaluates the baseline volume predictions. It can evaluate either the area baseline or the geometric baseline, depending on the --geom argument. It fits calibration functions on the training set, especially a quadratic mapping from baseline prediction to true volume, and applies this calibration to the test set. It then reports R², correlation, MAPE, and MAE, and saves plots comparing predicted and true spike volumes, including coloring by sampling date.

Run the file

```
metrics-baseline --output_area 'data/output_baseline' --input_geom '/path/to/dataset/baseline_jpg_ply_npz_with_stalk' --output_geom "data/output_geometric" --mapping_train 'data/path/to/mapping_train.json' --num_imgs 4 --geom True 
```

#### Neural Networks

Adapt main_training.py as described in the file and start training with 

```
poetry run accelerate launch --config_file single_gpu.yaml scripts/main_training.py --dataset-path /path/to/dataset/images_no_bar_crop --output-path data/folder_name --mapping-file /path/to/dataset/images_no_bar_crop/vol_mapping.csv 

```

Adapt evaluation.py to evaluate a specific model. 