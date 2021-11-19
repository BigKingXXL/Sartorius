from torch.utils.data import DataLoader
import warnings
from bigkingxxl.error.missing_property_error import MissingPropertyError

class Trainer:
    def __init__(self) -> None:
        super().__init__()
        self.__train_dataloader = None
        self.__test_dataloader = None
        self.__val_dataloader = None
    def train(self):
        self.__hasTrainDataloader()
        raise NotImplementedError
    def test(self):
        raise NotImplementedError
    def validate(self):
        raise NotImplementedError
    def setTrainingData(self, dataloader: DataLoader):
        self.__train_dataloader = dataloader
    def setTestData(self, dataloader: DataLoader):
        self.__test_dataloader = dataloader
    def setValidationData(self, dataloader: DataLoader):
        self.__val_dataloader = dataloader
    def __hasTrainDataloader(self, warn: bool = False):
        message = 'Train dataloader is not set.'
        if isinstance(self.__train_dataloader, None):
            if warn:
                warnings.warn(message)
            else:
                raise MissingPropertyError(message)
    def __hasTestDataloader(self, warn: bool = False):
        message = 'Test dataloader is not set.'
        if isinstance(self.__test_dataloader, None):
            if warn:
                warnings.warn(message)
            else:
                raise MissingPropertyError(message)
    def __hasValDataloader(self, warn: bool = False):
        message = 'Validation dataloader is not set.'
        if isinstance(self.__val_dataloader, None):
            if warn:
                warnings.warn(message)
            else:
                raise MissingPropertyError(message)
    