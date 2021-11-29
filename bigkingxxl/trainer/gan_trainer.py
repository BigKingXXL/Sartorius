import numpy as np
from torch._C import device
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Any, Callable
from logging import info
from sklearn.metrics import jaccard_score
from torch.utils.tensorboard import SummaryWriter
import os
import datetime

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
        discriminatorLoss: Any,
        device: str = 'cpu'
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
        self.__device = device
        self.__tensorboard_writer = SummaryWriter()
        self.__model_dir = f'models/{datetime.datetime.now()}'

    def train(self, epochs = 10):
        super(GanTrainer, self).hasTrainDataloader()
        info('starting training process')
        self.__generator.train()
        self.__discriminator.train()

        step = 0

        for epoch in range(1, epochs + 1):
            info(f'training epoch {epoch}')
            for inputImage, maskImage in self.train_dataloader:
                step += 1
                inputImage.to(self.__device)
                maskImage.to(self.__device)
                inputImage = inputImage[:, :512, :].reshape(-1, 1, 512, 704).float()
                maskImage = maskImage[:, :, :512,:].reshape(-1, 3, 512, 704).float()

                with self.__generatorFreezer:
                    # Train discriminator with generator
                    self.__discriminatorOptimizer.zero_grad()
                    predictedMask = self.__generator(inputImage)
                    discriminatorOutput = self.__discriminator(self.__addMask(inputImage, predictedMask))
                    loss = self.__discriminatorLoss(discriminatorOutput, torch.zeros_like(discriminatorOutput, device = discriminatorOutput.device))
                    loss.backward()
                    self.__discriminatorOptimizer.step()
                    self.__tensorboard_writer.add_scalar('Discriminator/Loss/Train/Real', loss, step)

                    # Train discriminator with labels
                    self.__discriminatorOptimizer.zero_grad()
                    predictedMask = self.__generator(inputImage)
                    discriminatorOutput = self.__discriminator(self.__addMask(inputImage, maskImage))
                    loss = self.__discriminatorLoss(discriminatorOutput, torch.ones_like(discriminatorOutput, device = discriminatorOutput.device))
                    loss.backward()
                    self.__discriminatorOptimizer.step()
                    self.__tensorboard_writer.add_scalar('Discriminator/Loss/Train/Generated', loss, step)
                
                with self.__discriminatorFreezer:
                    # Train generator
                    self.__generatorOptimizer.zero_grad()
                    predictedMask = self.__generator(inputImage)
                    discriminatorOutput = self.__discriminator(self.__addMask(inputImage, predictedMask))
                    loss = self.__discriminatorLoss(discriminatorOutput, torch.zeros_like(discriminatorOutput, device = discriminatorOutput.device))
                    loss.backward()
                    self.__generatorOptimizer.step()
                    loss = self.__generatorLoss(predictedMask, maskImage)
                    self.__tensorboard_writer.add_scalar('Generator/Loss/Train', loss, step)

            torch.save(self.__generator.state_dict(), os.path.join(self.__model_dir, f"epoch_{epoch}_generator.pth"))
            torch.save(self.__discriminator.state_dict(), os.path.join(self.__model_dir, f"epoch_{epoch}_discriminator.pth"))
            self.__tensorboard_writer.add_text("Training", f"Epoch {epoch} finished after {step} mini batches", epoch)

            with torch.no_grad():
                self.__generator.eval()
                self.__discriminator.eval()
                test_losses = []
                test_scores = []
                for inputImage, maskImage in self.test_dataloader:
                    inputImage.to(self.__device)
                    maskImage.to(self.__device)
                    inputImage = inputImage[:, :512, :].reshape(-1, 1, 520, 704).float()
                    maskImage = maskImage[:, :, :512,:].reshape(-1, 3, 520, 704).float()
                    predictedMask = self.__generator(inputImage)
                    score = jaccard_score(maskImage, predictedMask)
                    loss = self.__generatorLoss(predictedMask, maskImage)
                    test_losses.append(loss)
                    test_scores.append(score)
                self.__tensorboard_writer.add_scalar('Generator/Jaccard/Test', np.mean(score), epoch)
                self.__tensorboard_writer.add_scalar('Generator/Loss/Test', np.mean(test_losses), epoch)


    def __addMask(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.cat((image, mask.reshape(-1,3,512,704)), dim=1).to(device=self.__device)

    def test(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError
