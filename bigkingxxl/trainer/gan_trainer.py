import torch.nn as nn
import torch.optim as optim
import torch
from typing import Any, Callable
from logging import info

from bigkingxxl.trainer.trainer import Trainer

REAL_LABEL = 1
FAKE_LABEL = 0

class GanTrainer(Trainer):
    class ModelFreezer():
        def __init__(self, model: nn.Module) -> None:
            self.__model = model
        
        def __enter__(self):
            for param in self.__model.parameters():
                param.requires_grad = False
        
        def __exit__(self, type, value, traceback):
            for param in self.__model.parameters():
                param.requires_grad = True

    def __init__(self,
        generator: nn.Module,
        discriminator: nn.Module,
        generatorOptimizer: Callable[[nn.Module], optim.Optimizer],
        discriminatorOptimizer: Callable[[nn.Module], optim.Optimizer],
        generatorLoss: Any, # Pytorch loss functions have no supertype
        discriminatorLoss: Any
    ) -> None:
        super(GanTrainer).__init__()
        self.__generator = generator
        self.__discriminator = discriminator
        self.__generatorOptimizer = generatorOptimizer
        self.__discriminatorOptimizer = discriminatorOptimizer
        self.__generatorLoss = generatorLoss
        self.__discriminatorLoss = discriminatorLoss
        self.__discriminatorFreezer = self.ModelFreezer(discriminator)
        self.__generatorFreezer = self.ModelFreezer(generator)

    def train(self, epochs = 10):
        super(GanTrainer, self).hasTrainDataloader()
        info('starting training process')
        for epoch in range(1, epochs + 1):
            info(f'training epoch {epoch}')
            for inputImage, maskImage in self.train_dataloader:
                print(inputImage)
                print(maskImage)
                with self.__generatorFreezer:
                    # Train discriminator with generator
                    self.__discriminatorOptimizer.zero_grad()
                    predictedMask = self.__generator(inputImage)
                    discriminatorOutput = self.__discriminator(self.__addMask(inputImage, predictedMask))
                    loss = self.__discriminatorLoss(discriminatorOutput, torch.zeros_like(discriminatorOutput, device = discriminatorOutput.device))
                    loss.backward()
                    self.__discriminatorOptimizer.step(loss)

                    # Train discriminator with labels
                    self.__discriminatorOptimizer.zero_grad()
                    predictedMask = self.__generator(inputImage)
                    discriminatorOutput = self.__discriminator(self.__addMask(inputImage, maskImage))
                    loss = self.__discriminatorLoss(discriminatorOutput, torch.ones_like(discriminatorOutput, device = discriminatorOutput.device))
                    loss.backward()
                    self.__discriminatorOptimizer.step(loss)
                
                with self.__discriminatorFreezer:
                    # Train generator
                    self.__generatorOptimizer.zero_grad()
                    predictedMask = self.__generator(inputImage)
                    discriminatorOutput = self.__discriminator(self.__addMask(inputImage, predictedMask))
                    loss = self.__generatorLoss(discriminatorOutput, torch.zeros_like(discriminatorOutput, device = discriminatorOutput.device))
                    loss.backward()
                    self.__generatorOptimizer.step(loss)
                
    def __addMask(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.cat(image, mask, dim=1)

    def __freezeDiscriminator(self):
        return 

    def test(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError
