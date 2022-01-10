from typing import Optional
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data.dataloader import DataLoader

from bigkingxxl.dataset.dataset import SartoriusDataset, SartoriusDatasetPadded, SartoriusDatasetSquare, SartoriusDatasetUnscaled

class SartoriusDataLoader(LightningDataModule):
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = SartoriusDataset(mode='train')
            self.val_dataset = SartoriusDataset(mode='val')

        if stage in (None, "test"):
            self.test_dataset = SartoriusDataset(mode='val')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=1, num_workers=8)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=1, num_workers=8)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=1, num_workers=8)

class SartoriusDataLoaderUnscaled(LightningDataModule):
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = SartoriusDatasetUnscaled(mode='train')
            self.val_dataset = SartoriusDatasetUnscaled(mode='val')

        if stage in (None, "test"):
            self.test_dataset = SartoriusDatasetUnscaled(mode='val')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=1, num_workers=2)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=1, num_workers=2)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=1, num_workers=2)


class SartoriusDataLoaderSquare(LightningDataModule):
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = SartoriusDatasetSquare(mode='train')
            self.val_dataset = SartoriusDatasetSquare(mode='val')

        if stage in (None, "test"):
            self.test_dataset = SartoriusDatasetSquare(mode='val')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=1, num_workers=2)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=1, num_workers=2)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=1, num_workers=2)

class SartoriusDataLoaderPadded(LightningDataModule):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = SartoriusDatasetPadded(mode='train', height=self.height, width=self.width)
            self.val_dataset = SartoriusDatasetPadded(mode='val', height=self.height, width=self.width)

        if stage in (None, "test"):
            self.test_dataset = SartoriusDatasetPadded(mode='val', height=self.height, width=self.width)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=1, num_workers=8)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=1, num_workers=8)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=1, num_workers=8)
