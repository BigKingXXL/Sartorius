from torch.utils.data import Dataset
from torch import from_numpy, Tensor
import glob
import os
import cv2
from pickle import load as pickle_load

class SartoriusDataset(Dataset):
    def __init__(self, dataset_path: str = './dataset', mode = 'train'):
        self.__img_dir = os.path.join(dataset_path, mode)
        self.__masks_dir = os.path.join(dataset_path, 'masks')
        self.__ids = [el.split('.')[-2].split('/')[-1] for el in glob.glob(f'{self.__img_dir}/*.png')]

    def __len__(self) -> int:
        return len(self.__ids)

    def __getitem__(self, idx):
        selected = self.__ids[idx]
        image = cv2.imread(os.path.join(self.__img_dir, selected + '.png'), cv2.COLOR_BGR2GRAY)
        image_tensor = from_numpy(image)
        print(image_tensor.shape)
        with open(os.path.join(self.__masks_dir, selected + '.pickle'), 'rb') as read_file_handle:
            mask = pickle_load(read_file_handle)
        return image_tensor, mask