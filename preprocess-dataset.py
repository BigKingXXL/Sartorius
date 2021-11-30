import pandas as pd
import numpy as np
import pickle
import torch
import os

cell_types = ['astro', 'cort', 'shsy5y']

def convert_to_masks(path):
    with open(path) as read_handle:
        train_data = pd.read_csv(read_handle)
    for _, picture_group in train_data.groupby('id'):
        cells = {}

        picture_id = picture_group.iloc[0]['id']
        picture_size = picture_group.iloc[0]['width'] * picture_group.iloc[0]['height']

        for index, cell in picture_group.iterrows():
            cell_array = np.zeros(picture_size, dtype=np.int8)
            annotation = [int(val) for val in cell['annotation'].split(' ')]
            cell_type = cell['cell_type']

            for index in range(0, len(annotation), 2):
                cell_array[annotation[index]:annotation[index]+annotation[index+1]].fill(1)

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
            result_tensor[cell_index, :, :,] = torch.from_numpy(sum_mask).view(size=(520, 704,))

        if result_tensor.isnan().any():
            input("Failure")

        os.makedirs(os.path.join('./dataset', 'masks'), exist_ok=True)
        torch.save(result_tensor, os.path.join('./dataset', 'masks', f'{picture_id}.tensor'))

if __name__ == '__main__':
    convert_to_masks('./dataset/train.csv')