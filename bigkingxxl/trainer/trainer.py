from torch.utils.data import DataLoader
import warnings
from bigkingxxl.error.missing_property_error import MissingPropertyError

class Trainer:
    def __init__(self) -> None:
        super().__init__()
        self.train_dataloader = None
        self.test_dataloader = None
        self.val_dataloader = None

    def train(self):
        self.hasTrainDataloader()
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def setTrainingData(self, dataloader: DataLoader):
        self.train_dataloader = dataloader

    def setTestData(self, dataloader: DataLoader):
        self.test_dataloader = dataloader

    def setValidationData(self, dataloader: DataLoader):
        self.val_dataloader = dataloader

    def hasTrainDataloader(self, warn: bool = False):
        message = 'Train dataloader is not set.'
        if self.train_dataloader is None:
            if warn:
                warnings.warn(message)
            else:
                raise MissingPropertyError(message)

    def hasTestDataloader(self, warn: bool = False):
        message = 'Test dataloader is not set.'
        if self.test_dataloader is None:
            if warn:
                warnings.warn(message)
            else:
                raise MissingPropertyError(message)

    def hasValDataloader(self, warn: bool = False):
        message = 'Validation dataloader is not set.'
        if self.val_dataloader is None:
            if warn:
                warnings.warn(message)
            else:
                raise MissingPropertyError(message)
    