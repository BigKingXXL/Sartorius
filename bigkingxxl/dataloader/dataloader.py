from typing import Optional
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data.dataloader import DataLoader

from bigkingxxl.dataset.dataset import SartoriusDataset

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
        return DataLoader(self.train_dataset, batch_size=1)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=1)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=1)
