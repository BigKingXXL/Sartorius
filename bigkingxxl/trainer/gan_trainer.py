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
import re

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
        self.generator = generator
        self.discriminator = discriminator
        self.generatorOptimizer = generatorOptimizer
        self.discriminatorOptimizer = discriminatorOptimizer
        self.generatorLoss = generatorLoss
        self.discriminatorLoss = discriminatorLoss
        self.discriminatorFreezer = self.ModelFreezer(discriminator)
        self.generatorFreezer = self.ModelFreezer(generator)
        self.device = device
        self.tensorboard_writer = SummaryWriter()
        self.model_dir = re.sub(r'[\ \-\.\:]', '_', f'models/{datetime.datetime.now()}')
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, epochs = 10):
        super(GanTrainer, self).hasTrainDataloader()
        info('starting training process')
        self.generator.train()
        self.discriminator.train()

        step = 0

        for epoch in range(1, epochs + 1):
            info(f'training epoch {epoch}')
            for inputImage, maskImage in self.train_dataloader:
                print(step)
                
                # torch.cuda.empty_cache()
                step += 1
                inputImage = inputImage[:, :512, :].reshape(-1, 1, 512, 704).float().to(self.device)
                maskImage = maskImage[:, :, :512,:].reshape(-1, 3, 512, 704).float().to(self.device)

                # with self.generatorFreezer:
                #     # Train discriminator with generator
                #     self.discriminatorOptimizer.zero_grad()
                #     predictedMask = self.generator(inputImage)
                #     discriminatorOutput = self.discriminator(self.addMask(inputImage, predictedMask))
                #     loss = self.discriminatorLoss(discriminatorOutput, torch.zeros_like(discriminatorOutput, device = discriminatorOutput.device))
                #     loss.backward()
                #     self.discriminatorOptimizer.step()
                #     self.tensorboard_writer.add_scalar('Discriminator/Loss/Train/Real', loss.to('cpu'), step)

                #     # Train discriminator with labels
                #     self.discriminatorOptimizer.zero_grad()
                #     predictedMask = self.generator(inputImage)
                #     discriminatorOutput = self.discriminator(self.addMask(inputImage, maskImage))
                #     loss = self.discriminatorLoss(discriminatorOutput, torch.ones_like(discriminatorOutput, device = discriminatorOutput.device))
                #     loss.backward()
                #     self.discriminatorOptimizer.step()
                #     self.tensorboard_writer.add_scalar('Discriminator/Loss/Train/Generated', loss.to('cpu'), step)
                
                with self.discriminatorFreezer:
                    # Train generator
                    self.generatorOptimizer.zero_grad()
                    predictedMask = self.generator(inputImage)
                    #discriminatorOutput = self.discriminator(self.addMask(inputImage, predictedMask))
                    #loss = self.discriminatorLoss(discriminatorOutput, torch.zeros_like(discriminatorOutput, device = discriminatorOutput.device))
                    #loss.backward()
                    #self.generatorOptimizer.step()
                    loss = self.generatorLoss(predictedMask, maskImage)
                    loss.backward()
                    self.generatorOptimizer.step()
                    self.tensorboard_writer.add_scalar('Generator/Loss/Train', loss.to('cpu'), step)

            torch.save(self.generator.state_dict(), os.path.join(self.model_dir, f"epoch_{epoch}_generator.pth"))
            torch.save(self.discriminator.state_dict(), os.path.join(self.model_dir, f"epoch_{epoch}_discriminator.pth"))
            self.tensorboard_writer.add_text("Training", f"Epoch {epoch} finished after {step} mini batches", epoch)

            with torch.no_grad():
                self.generator.eval()
                self.discriminator.eval()
                test_losses = []
                test_scores = []
                for inputImage, maskImage in self.test_dataloader:
                    inputImage = inputImage[:, :512, :].reshape(-1, 1, 512, 704).float().to(self.device)
                    maskImage = maskImage[:, :, :512,:].reshape(-1, 3, 512, 704).float().to(self.device)
                    predictedMask = self.generator(inputImage)
                    score = jaccard_score(maskImage.cpu().detach().numpy().reshape(-1), self.to_binary(predictedMask.cpu().detach()).numpy().reshape(-1))
                    loss = self.generatorLoss(predictedMask, maskImage)
                    test_losses.append(loss.cpu())
                    test_scores.append(score)
                self.tensorboard_writer.add_scalar('Generator/Jaccard/Test', np.mean(score), epoch)
                self.tensorboard_writer.add_scalar('Generator/Loss/Test', np.mean(test_losses), epoch)


    def addMask(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.cat((image, mask.reshape(-1,3,512,704)), dim=1).to(device=self.device)

    def test(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def to_binary(self, tensor: torch.Tensor, threshold = 0.5) -> torch.Tensor:
        return (tensor > threshold).float()
