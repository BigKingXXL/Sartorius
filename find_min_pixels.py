import pandas as pd
import numpy as np
import logging


cell_types = ['astro', 'cort', 'shsy5y']

def convert_to_masks(path):
    with open(path) as read_handle:
        train_data = pd.read_csv(read_handle)
        cells = {}
        cellSize = {}

        # picture_id = train_data.iloc[0]['id']
        picture_size = train_data.iloc[0]['width'] * train_data.iloc[0]['height']
        for index, cell in train_data.iterrows():
            cell_array = np.zeros(picture_size, dtype=np.int8)
            annotation = [int(val) for val in cell['annotation'].split(' ')]
            cell_type = cell['cell_type']

            for index in range(0, len(annotation), 2):
                cell_array[annotation[index] - 1:annotation[index]+annotation[index+1] - 2].fill(1)

            if not cell_type in cells:
                cells[cell_type] = []
                cellSize[cell_type] = []
            cells[cell_type].append(cell_array)
            cellSize[cell_type].append(cell_array.sum())
        return cellSize

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("converting masks to seperated layers")
    output = convert_to_masks('./dataset/train.csv')
    for cell_type, liste in output.items():
        print(cell_type, np.min(liste))
