# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from os.path import join, exists
import os
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    #project_root = join("..", "..")
    #raw_data_dir = join(project_root, "data", "raw")
    #raw_data_file = join(raw_data_dir, "breast-cancer-wisconsin.data")

    column_names = [
        "Sample",
        "Clump Thickness",
        "Cell Size Uniformity",
        "Cell Shape Uniformity",
        "Marginal Adhesion",
        "Single Epithelial Cell Size",
        "Bare Nuclei",
        "Bland Chromatin", 
        "Normal Nuclei",
        "Mitoses",
        "Class",
    ]

    df = pd.read_csv(input_filepath, names=column_names, index_col=0, 
                     na_values=["?"])
    
    df_clean = df.dropna()

    # Convert the column "Bare Nuclei" to an int
    df_clean['Bare Nuclei'] = df_clean['Bare Nuclei'].astype(np.int64)

    # Verify path exists
    path = join("data", "processed")
    if not(exists(path)):
        # Try to create path
        os.makedirs(path)

    # Save the file as a csv
    df_clean.to_csv(output_filepath)

    logger.info(f"processed data saved to {output_filepath}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
