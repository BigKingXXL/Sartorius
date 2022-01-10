from argparse import ArgumentParser
import time
from logging import log
from numpy.core.fromnumeric import argmax
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch._C import _set_default_mobile_cpu_allocator
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional

from bigkingxxl.utils.cupy_tensor import tensorToCupy, cupyToTensor

if torch.cuda.is_available():
    import cupy as np
    from cucim.skimage.color import gray2rgb
else:
    import numpy as np
    from skimage.color import gray2rgb

from torch.utils.data.dataloader import DataLoader
from bigkingxxl.evaluator.evalutator import iou_map, label_instances
from bigkingxxl.loss.IoULoss import IoULoss
import torchvision

from bigkingxxl.trainer.trainer import Trainer

REAL_LABEL = 1
FAKE_LABEL = 0

class GanTrainer(LightningModule):
    def __init__(self,
        generator: nn.Module,
        discriminator: nn.Module,
        data_module: LightningDataModule,
        lr: float,
        b1: float,
        b2: float,
        threshold: float,
        channels: int = 3,
        width: int = 520,
        height: int = 704,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator", "width", "height"])
        data_shape = (channels, width, height)
        self.generator = generator
        self.discriminator = discriminator
        self.loss = nn.CrossEntropyLoss()
        self.data_module = data_module
        # self.model_dir = re.sub(r'[\ \-\.\:]', '_', f'models/{datetime.datetime.now()}')
        # os.makedirs(self.model_dir, exist_ok=True)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("GanModel")
        parser.add_argument("--lr", type=float, default=0.0005)
        parser.add_argument("--b1", type=float, default=0.9)
        parser.add_argument("--b2", type=float, default=0.999)
        parser.add_argument("--threshold", type=float, default=0.5)
        return parent_parser

    def adverserial_loss(self, y, y_hat):
        return self.loss(y, y_hat)

    def forward(self, X):
        return self.generator(X)
    
    def binarize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        threshold = self.hparams.threshold
        return (mask > threshold).type_as(mask)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, real_masks = batch

        fake = torch.empty((imgs.size(0), 1)).fill_(FAKE_LABEL)
        fake = fake.type_as(imgs)

        valid = torch.empty((imgs.size(0), 1)).fill_(REAL_LABEL)
        valid = valid.type_as(imgs)

        # Train generator
        if optimizer_idx == 0:
            if not (np.random.choice((True, False), size=1, p=(0.2, 0.8))):
                return None
            generated_masks = self.generator(imgs)
            binary_generated_mask = generated_masks # self.binarize_mask(generated_masks)
            discriminator_output = self.discriminator(self.addMask(imgs, binary_generated_mask)).mean()
            generator_loss = -discriminator_output
            tqdm_dict = {"g_loss": float(generator_loss)}
            self.log_dict(tqdm_dict, prog_bar=True)
            return generator_loss

        # Train discriminators
        if optimizer_idx != 0:
            self.discriminator
            generated_masks = self.generator(imgs)
            binary_generated_mask = generated_masks # self.binarize_mask(generated_masks)
            discriminator_loss_fake = self.discriminator(self.addMask(imgs, binary_generated_mask)).mean()
            discriminator_loss_real = self.discriminator(self.addMask(imgs, real_masks)).mean()
            discriminator_loss =  (discriminator_loss_fake - discriminator_loss_real)
            tqdm_dict = {"d_loss": float(discriminator_loss)}
            self.log_dict(tqdm_dict, prog_bar=True)
            return discriminator_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        optimizer_generator = torch.optim.RMSprop(self.generator.parameters(), lr=lr)
        optimizer_discriminator = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr)
        return [ optimizer_generator, optimizer_discriminator ], []
    
    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        imgs, real_masks = batch
        generated_masks = self.generator(imgs)
        binary_generated_masks = tensorToCupy(self.binarize_mask(generated_masks))
        generated_instances = label_instances(binary_generated_masks)
        combined_instance_masks = self.combine_instances(generated_instances, generated_masks)
        iou_score = iou_map(list(combined_instance_masks), list(tensorToCupy(real_masks)))
        self.log_dict({'val_iou': iou_score.tolist()}, prog_bar=True)
    
    def combine_instances(self, instance_masks: np.ndarray, logits: torch.Tensor) -> np.ndarray:
        """Combines a batch of instances of different cell-types into one layer of instances.
        Removes overlaps by using the highest logit found in the logits tensor.

        Args:
            instance_masks (torch.Tensor): Tensor of size (batch size, number of cell-types, width, height) with numbered instances in each layer.
            logits (torch.Tensor): Tensor of size (batch size, number of cell-types, width, height) with logits.

        Returns:
            torch.Tensor: Tensor of size (batch size, width, height) with numbered instances.
        """
        combined_instances = np.zeros((instance_masks.shape[0], instance_masks.shape[2], instance_masks.shape[3]))
        for batch_index, batch in enumerate(instance_masks):
            max_layers = [0] + [int(batch[layer, :, :].max()) for layer in range(batch.shape[0])][:-1]    
            for index in range(1, len(max_layers)):
                max_layers[index] += max_layers[index - 1]
                instance_masks[batch_index][index] += max_layers[index]

        for batch_index, batch in enumerate(logits):
            maximum = tensorToCupy(torch.argmax(batch, dim=0))
            combined_instances[batch_index] = np.take_along_axis(instance_masks[batch_index], np.array([maximum]), axis=0)
        return combined_instances

    def addMask(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.cat((image, mask.reshape(-1,3,512,704)), dim=1).type_as(image)
    
    def on_epoch_end(self) -> None:
        X, y = self.data_module.train_dataloader().dataset[0]
        X = X.type_as(self.discriminator.net[2].weight)
        sample_imgs = self.generator(X.unsqueeze_(dim=0))
        grid = torchvision.utils.make_grid([cupyToTensor(gray2rgb(tensorToCupy(X.squeeze_(dim=0)[0]))).permute(2, 0, 1).cpu(), sample_imgs.squeeze_(dim=0).cpu(), y.cpu()], nrow=3)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

