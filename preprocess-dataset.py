from glob import glob
import pandas as pd
import numpy as np
import torch
import os
import tiffile
from PIL import Image
import logging


cell_types = ['astro', 'cort', 'shsy5y']

def convert_to_celltype_masks(path):
    """
    This method creates the cell masks. Each mask contains all the cell masks for one image.
    We create a 3D tensor which consists of 2D binary cell masks for every cell type.

    Parameters:
        path (str): Path to the .csv file that contains all the cell annotations.
    """

    with open(path) as read_handle:
        train_data = pd.read_csv(read_handle)
    
    # We group by the picture id to get all the annotations for one image.
    # Reminder: The annotation for each cell is encoded in one row.
    for _, picture_group in train_data.groupby('id'):
        cells = {}
        picture_id = picture_group.iloc[0]['id']
        picture_size = picture_group.iloc[0]['width'] * picture_group.iloc[0]['height']

        for index, cell in picture_group.iterrows():
            # We start by creating a 1D array in which we will save the annotation for a single cell.
            cell_array = np.zeros(picture_size, dtype=np.int8)
            annotation = [int(val) for val in cell['annotation'].split(' ')]
            cell_type = cell['cell_type']

            for index in range(0, len(annotation), 2):
                cell_array[annotation[index] - 1:annotation[index]+annotation[index+1] - 2].fill(1)

            # The "cells" dictionary consists of a list of 2D annotation masks for every cell type.
            if not cell_type in cells:
                cells[cell_type] = []
            cells[cell_type].append(cell_array)
        
        # Now we need to combine all individual masks
        result_tensor = torch.zeros((3, 520, 704,))
        for cell_index, cell_type in enumerate(cells):

            if not cell_type in cells:
                continue

            mask_arrays = cells[cell_type]

            sum_mask = mask_arrays[0]

            # This is done by summing up all masks of one cell type.
            for mask in mask_arrays[1:]:
                sum_mask += mask
            
            # We added the feature of overlap removal.
            # Although this is not needed for the given train labels, this method can also be called on modle outputs.
            # Therefore this ensures that model predictions also do not produce overlapping labels.
            sum_mask = np.where((sum_mask <= 1) & (sum_mask >= 0), sum_mask, 0)
            result_tensor[cell_index, :, :,] = torch.from_numpy(sum_mask).reshape((520, 704,))

        if result_tensor.isnan().any():
            input("Failure")

        os.makedirs(os.path.join('./dataset', 'masks', 'train'), exist_ok=True)
        torch.save(result_tensor, os.path.join('./dataset', 'masks', 'train' , f'{picture_id}.tensor'))

def convert_to_val_masks(path):
    """
    This method creates the cell masks. Each mask contains all the cell masks for one image.
    We assign every individual cell one unique cell id.

    Parameters:
        path (str): Path to the .csv file that contains all the cell annotations.
    """

    with open(path) as read_handle:
        train_data = pd.read_csv(read_handle)
        
    # We group by the picture id to get all the annotations for one image.
    # Reminder: The annotation for each cell is encoded in one row.
    for _, picture_group in train_data.groupby('id'):

        picture_id = picture_group.iloc[0]['id']
        picture_size = picture_group.iloc[0]['width'] * picture_group.iloc[0]['height']

        # We now create a 1D array where we save the annotations for each cell mask.
        cell_array = np.zeros(picture_size, dtype=int)
        # We assign each cell a cell id starting from one (The backround has id zero).
        for cell_index, (_, cell) in enumerate(picture_group.iterrows()):
            annotation = [int(val) for val in cell['annotation'].split(' ')]
            for index in range(0, len(annotation), 2):
                # We now fill the corresponting mask of each cell with its id.
                cell_array[annotation[index] - 1:annotation[index]+annotation[index+1] - 2].fill(cell_index + 1)

        # To make things easier, we previously used a 1D array, which we now need to reshape into a 2D array.
        result_tensor = torch.from_numpy(cell_array.reshape(520, 704))

        # Now we just need to save our results.
        os.makedirs(os.path.join('./dataset', 'masks', 'val'), exist_ok=True)
        os.makedirs(os.path.join('./dataset', 'tif', 'train'), exist_ok=True)
        torch.save(result_tensor, os.path.join('./dataset', 'masks', 'val' , f'{picture_id}.tensor'))
        tiffile.imsave(os.path.join('./dataset', 'tif', 'train', f'{picture_id}_masks.tif'), result_tensor.numpy().astype(np.uint8))

def convert_png_images_to_tiff(path: str, subfolder: str) -> None:
    """
    This method converts the competition images which are given as .png files into .tiff files.
    These files are required by cellpose.

    Parameters:
        path (str): The path to the directory where the png images are located.
        subfolder (str): A name for the new subdirectory in which the tif files are saved, e.g. train, val, test.
    """

    basepath = os.path.join('./dataset', 'tif', subfolder)
    os.makedirs(basepath, exist_ok=True)
    for file in glob(os.path.join(f"{path}", "*.png")):
        img = Image.open(file)
        picture_id = file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        tiffile.imsave(os.path.join(basepath, f'{picture_id}.tif'), np.array(img))
        img.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("converting masks to seperated layers")
    convert_to_celltype_masks('./dataset/train.csv')
    logging.info("converting masks to one layers")
    convert_to_val_masks('./dataset/train.csv')
    logging.info("converting input images to tif")
    convert_png_images_to_tiff('./dataset/train', 'train')
    convert_png_images_to_tiff('./dataset/test', 'test')
    logging.info("done")
