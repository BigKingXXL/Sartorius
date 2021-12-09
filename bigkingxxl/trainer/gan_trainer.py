from argparse import ArgumentParser
from logging import log
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional
from bigkingxxl.evaluator.evalutator import iou_map, label_instances

from bigkingxxl.trainer.trainer import Trainer

REAL_LABEL = 1
FAKE_LABEL = 0

class GanTrainer(LightningModule):
    def __init__(self,
        generator: nn.Module,
        discriminator: nn.Module,
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
        # self.model_dir = re.sub(r'[\ \-\.\:]', '_', f'models/{datetime.datetime.now()}')
        # os.makedirs(self.model_dir, exist_ok=True)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("GanModel")
        parser.add_argument("--lr", type=float, default=0.0001)
        parser.add_argument("--b1", type=float, default=0.9)
        parser.add_argument("--b2", type=float, default=0.999)
        parser.add_argument("--threshold", type=float, default=0.4)
        return parent_parser

    def adverserial_loss(self, y, y_hat):
        return F.binary_cross_entropy(y, y_hat)

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
            generated_masks = self.generator(imgs)
            binary_generated_mask = generated_masks # self.binarize_mask(generated_masks)
            discriminator_output = self.discriminator(self.addMask(imgs, binary_generated_mask))
            generator_loss = self.adverserial_loss(discriminator_output, fake)
            tqdm_dict = {"g_loss": float(generator_loss)}
            output = {"loss": generator_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            return output

        # Train discriminators
        if optimizer_idx == 1:
            generated_masks = self.generator(imgs)
            binary_generated_mask = generated_masks # self.binarize_mask(generated_masks)
            discriminator_loss_fake = self.adverserial_loss(self.discriminator(self.addMask(imgs, binary_generated_mask)), fake)
            discriminator_loss_real = self.adverserial_loss(self.discriminator(self.addMask(imgs, real_masks)), valid)
            discriminator_loss = (discriminator_loss_fake + discriminator_loss_real) / 2
            tqdm_dict = {"d_loss": float(discriminator_loss)}
            output = {"loss": discriminator_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [ optimizer_generator, optimizer_discriminator ], []
    
    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        imgs, real_masks = batch
        generated_masks = self.generator(imgs)
        binary_generated_masks = self.binarize_mask(generated_masks).numpy()
        generated_instances = label_instances(binary_generated_masks)
        combined_instance_masks = self.combine_instances(generated_instances, generated_masks)
        iou_score = iou_map(list(combined_instance_masks), list(real_masks.numpy()))
        self.log_dict({'val_iou': iou_score})
    
    def combine_instances(self, instance_masks: np.ndarray, logits: np.ndarray) -> torch.Tensor:
        """Combines a batch of instances of different cell-types into one layer of instances.
        Removes overlaps by using the highest logit found in the logits tensor.

        Args:
            instance_masks (torch.Tensor): Tensor of size (batch size, number of cell-types, width, height) with numbered instances in each layer.
            logits (torch.Tensor): Tensor of size (batch size, number of cell-types, width, height) with logits.

        Returns:
            torch.Tensor: Tensor of size (batch size, width, height) with numbered instances.
        """
        masked_logits = logits
        combined_instances = np.zeros((instance_masks.shape[0], instance_masks.shape[2], instance_masks.shape[3]))
        for batch in range(instance_masks.shape[0]):
            layer_numbers = [0] + [int(instance_masks[batch, layer, :, :].max()) for layer in range(instance_masks.shape[1])][:-1]
            for x in range(instance_masks.shape[2]):
                for y in range(instance_masks.shape[3]):
                    max_layer = 0
                    max_value = masked_logits[batch, 0, x, y] if masked_logits[batch, 0, x, y] > self.hparams.threshold else 0
                    for layer in range(1, instance_masks.shape[1]):
                        if masked_logits[batch, layer, x, y] > max_value and masked_logits[batch, layer, x, y] > self.hparams.threshold:
                            max_layer = layer
                            max_value = masked_logits[batch, layer, x, y]
                    combined_instances[batch, x, y] = 0 if max_value == 0 else instance_masks[batch, max_layer, x, y] + layer_numbers[max_layer]
        return combined_instances

    def addMask(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.cat((image, mask.reshape(-1,3,512,704)), dim=1).type_as(image)

