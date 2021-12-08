from typing import Tuple
import torch
from torch.utils.data import Dataset
from torch import from_numpy, Tensor
import glob
import os
import cv2

class SartoriusDataset(Dataset):
    def __init__(self, dataset_path: str = './dataset', mode = 'train', trainsplit = 0.8, preprocess: bool = False) -> None:
        self.img_dir = os.path.join(dataset_path, 'train')
        self.train_masks_dir = os.path.join(dataset_path, 'masks', 'train')
        self.val_masks_dir = os.path.join(dataset_path, 'masks', 'val')
        self.masks_dir = self.train_masks_dir if mode == 'train' else self.val_masks_dir
        self.ids = sorted([el.split('.')[-2].split('/')[-1] for el in glob.glob(f'{self.img_dir}/*.png')])
        if mode == 'train':
            self.ids = self.ids[:int(trainsplit*len(self.ids))]
        elif mode == 'val':
            self.ids = self.ids[int((1-trainsplit)*len(self.ids)):]
        else:
            raise Exception("Unknown mode")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        selected = self.ids[idx]
        image = cv2.imread(os.path.join(self.img_dir, selected + '.png'), cv2.COLOR_BGR2GRAY)
        image_tensor = from_numpy(image)[:512, :].reshape(1, 512, 704).float()
        mask = torch.load(os.path.join(self.masks_dir, selected + '.tensor'))
        if len(mask.shape) == 2:
            mask = mask[:512, :]
        else:
            mask = mask[:, :512, :]
        return image_tensor, mask