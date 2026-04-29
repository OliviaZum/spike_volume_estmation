import pandas
import pandas as pd
import re
import pathlib
import shutil
import os
import click
import csv 
from pathlib import Path
from icecream import ic

def debug_csv(input_path : Path):

    """
    Look at all the csv entries and check how many images per wheat plant
    we have.

    :return: None
    """

    input_path = input
    csv_df = []

    for filename in os.listdir(input_path): 
        if filename.endswith(".csv") and not filename.startswith("._"):  # Check if the file is a CSV file
            print(filename)
            csv_df.append(pandas.read_csv(filename, encoding="utf-8-sig"))

    all_csv = pd.concat(csv_df)
    all_csv.groupby(["row_lot", "range_lot", "plant_id"]).count().to_csv("groups.csv")

def create_mapping(input_path : Path, volume_path : Path, output_path : Path):

    """
    Create mapping from images to volume
    Reads the volume form txt files and collects all data in csv file.
    Finally the csv file get stored in the same folder as this file.

    :return:
    """

    csv_df = []
    input_csv_path = input_path
    error_file_path = output_path / "error.txt"
    if error_file_path.exists():
        error_file_path.unlink()

    # make one large pandas file with the four data files we have
    for filename in os.listdir(input_csv_path): 
        if filename.endswith(".csv"): 
            file_path = os.path.join(input_path, filename)
            csv_df.append(pandas.read_csv(file_path, encoding="utf-8-sig"))

    all_csv = pd.concat(csv_df)
    all_csv = all_csv[["row_lot", "range_lot", "plant_id", "number"]]
    all_csv.drop_duplicates()

    # specify the location of the text files containing the volume data
    vol_dir = volume_path

    # create dataframes to collect data
    mapping_df = pd.DataFrame(columns=["img_id", "plant_id", "volume", "img_name"])
    not_found_df = pd.DataFrame(columns=["plant_id"])

    count = 0
    not_count = 0
    for i, row in all_csv.iterrows():
        x = row["row_lot"]
        y = row["range_lot"]
        z = row["plant_id"]
        n = row["number"]
        vol_path = pathlib.Path(vol_dir, f"{y}_{x}_{z}.txt")
        if os.path.exists(vol_path):
            # if found, then add to mapping
            f = open(vol_path)
            volume = float(f.read())
            f.close()
            new_entry = {"img_id": f"{y}_{x}_{z}_{n}", "plant_id": f"{y}_{x}_{z}",
                         "volume": volume, "img_name": f"{y}_{x}_{z}_{n}.jpg" }
            mapping_df.loc[len(mapping_df)] = new_entry
            count += 1
        else:
            # if not found log in other file
            print(f"\nPlant not found {y}_{x}_{z} volume")
            not_found_df.loc[len(not_found_df)] = {"plant_id": f"{y}_{x}_{z}"}
            not_count += 1

            with (error_file_path).open(mode="a") as f:
                    f.write(f"{y}_{x}_{z}\n")
                    f.close()


    not_found_df.drop_duplicates()
    print(f"Count : {count}, not found : {not_count}")

    # create output files
    mapping_df = mapping_df.drop_duplicates(subset='img_name', keep='first')
    mapping_df = mapping_df.loc[:, ~mapping_df.columns.str.startswith('Unnamed')]

    mapping_df.to_csv(output_path / "vol_mapping.csv", index=False)
    not_found_df.to_csv(output_path / "not_found_vol.csv", index=False)


def copy_imgs(input_path : Path, img_path : Path, output_path : Path):

    """
    Copies all the images which are assigned to some location on the field
    in the document at csv_path to a new folder with corresponding names
    name.

    :param csv_path: Path to file containing information about the image
    :param imgs_path: Path to the location where the images are stored currently
    :return:
    """

    new_path = output_path

    # make one large pandas file with the four data files we have
    for filepath in input_path.iterdir():
        filename = filepath.name
        if filepath.suffix == ".csv" and not filename.startswith("._"):  # Check if the file is a CSV file
            
            print(f"\n\nStart copying file {filepath}")
            #pattern = r'..WW.*$'
            #pattern = r'E.*$'
            #pattern = r'C.*$'
            pattern = r'\d{1,3}_Picture.*$'

            excel_df = pd.read_csv(filepath)
            count = 0
            
            
            for i, row in excel_df.iterrows():
                count += 1
                x = row["row_lot"]
                y = row["range_lot"]
                z = row["plant_id"]
                n = row["number"]
                val = row["value"]
                
                try: 
                    name = re.findall(pattern, val)[0]
                    print(name)
                    new_name = f"{y}_{x}_{z}_{n}.jpg"
                    source = pathlib.Path(img_path, name)
                    target = pathlib.Path(new_path, new_name)

                    if target.exists():
                        print("image already renamed")
                        continue

                    shutil.copy(source, target)
                    if i % 50 == 0:
                        print(f"\nFiles processed: {i} --current img {new_name}")
                except (IndexError, FileNotFoundError) as e:
                    print(f"skipping row {i}: Image path or name not found for val '{val}'")
                    continue
                    
            print(f"\n\nProcessed {filepath} images total: {count}")

@click.command()
@click.option("--input", required=True, type=str, help="Path to the input data folder containing the .csv files")
@click.option("--volume", required=True, type=str, help="Path to the volume files containing the .txt files")
@click.option("--imgs", required=True, type=str, help="Path to the location where the images are stored currently") #"../../data/1_3D/Output/Jonas_final_{i}/Picture/"
@click.option("--output", required=True, type=str, help="Path to the output data folder")
def main(input, volume, imgs, output):
    input_path = Path(input)
    volume_path = Path(volume)
    img_path = Path(imgs)
    output_path = Path(output)

    print("input path :", input_path.exists())
    copy_imgs(input_path, img_path, output_path)
    create_mapping(input_path, volume_path, output_path)


if __name__ == "__main__":
    main()