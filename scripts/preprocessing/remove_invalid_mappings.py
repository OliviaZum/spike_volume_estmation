import pandas as pd
from pathlib import Path
from icecream import ic
import click
import numpy as np


@click.command()
@click.option('--mapping-file', required=True, type=str, help='Mapping file to remove columns with volume == -1')
def main(mapping_file : str):

    """Removes all rows with -1 volume form the input `mapping_file`

    Args:
        mapping_file (str): File with mappings
    """

    mapping_file = Path(mapping_file)
    
    if not mapping_file.exists() or not mapping_file.is_file():
        print(f'Mapping file does not exists! Choose different file.')
        return

    df_mappings = pd.read_csv(mapping_file)
    
    if not 'volume' in df_mappings.columns:
        print(f'Mapping file does not contain volume column.')
        return
    
    df_mappings.drop(df_mappings.index[df_mappings.loc[:, 'volume'] == -1], inplace=True)

    #remove potential dublications
    df_mappings = df_mappings.drop_duplicates(subset='img_id', keep='first')
    df_mappings = df_mappings.loc[:, ~df_mappings.columns.str.startswith('Unnamed')]
    df_mappings.to_csv(mapping_file, index=False)
    
    