from glob import glob
import pandas as pd
import numpy as np
import pickle
import torch
import os
import tiffile
from PIL import Image
import logging


cell_types = ['astro', 'cort', 'shsy5y']

def convert_to_masks(path):
    with open(path) as read_handle:
        train_data = pd.read_csv(read_handle)
    for _, picture_group in train_data.groupby('id'):
        cells = {}

        picture_id = picture_group.iloc[0]['id']
        picture_size = picture_group.iloc[0]['width'] * picture_group.iloc[0]['height']

        cell_array = np.zeros(picture_size, dtype=np.int8)
        for index, cell in picture_group.iterrows():
            annotation = [int(val) for val in cell['annotation'].split(' ')]
            cell_type = cell['cell_type']

            for index in range(0, len(annotation), 2):
                cell_array[annotation[index] - 1:annotation[index]+annotation[index+1] - 2].fill(1)

            if not cell_type in cells:
                cells[cell_type] = []
            cells[cell_type].append(cell_array)
        

        result_tensor = torch.zeros((3, 520, 704,))
        for cell_index, cell_type in enumerate(cells):

            if not cell_type in cells:
                print("Filling")
                # result_tensor[cell_index, :, :,].fill_(0)
                continue

            mask_arrays = cells[cell_type]

            sum_mask = mask_arrays[0]

            for mask in mask_arrays[1:]:
                sum_mask += mask
            
            # REMOVE OVERLAP
            # TODO: MAKE ME FANCY
            sum_mask = np.where((sum_mask <= 1) & (sum_mask >= 0), sum_mask, 0)
            result_tensor[cell_index, :, :,] = torch.from_numpy(sum_mask).reshape((520, 704,))

        if result_tensor.isnan().any():
            input("Failure")

        os.makedirs(os.path.join('./dataset', 'masks', 'train'), exist_ok=True)
        torch.save(result_tensor, os.path.join('./dataset', 'masks', 'train' , f'{picture_id}.tensor'))

def convert_to_val_masks(path):
    with open(path) as read_handle:
        train_data = pd.read_csv(read_handle)
        
    for _, picture_group in train_data.groupby('id'):

        picture_id = picture_group.iloc[0]['id']
        picture_size = picture_group.iloc[0]['width'] * picture_group.iloc[0]['height']

        cell_array = np.zeros(picture_size, dtype=int)
        for cell_index, (_, cell) in enumerate(picture_group.iterrows()):
            annotation = [int(val) for val in cell['annotation'].split(' ')]
            for index in range(0, len(annotation), 2):
                cell_array[annotation[index] - 1:annotation[index]+annotation[index+1] - 2].fill(cell_index + 1)

        result_tensor = torch.from_numpy(cell_array.reshape(520, 704))
        os.makedirs(os.path.join('./dataset', 'masks', 'val'), exist_ok=True)
        os.makedirs(os.path.join('./dataset', 'tif', 'train'), exist_ok=True)
        torch.save(result_tensor, os.path.join('./dataset', 'masks', 'val' , f'{picture_id}.tensor'))
        tiffile.imsave(os.path.join('./dataset', 'tif', 'train', f'{picture_id}_masks.tif'), result_tensor.numpy().astype(np.uint8))
        # print(result_tensor.numpy().astype(np.uint8).max())

def convert_png_images_to_tiff(path: str, subfolder: str) -> None:
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
    convert_to_masks('./dataset/train.csv')
    logging.info("converting masks to one layers")
    convert_to_val_masks('./dataset/train.csv')
    logging.info("converting input images to tif")
    convert_png_images_to_tiff('./dataset/train', 'train')
    convert_png_images_to_tiff('./dataset/test', 'test')
    logging.info("done")
