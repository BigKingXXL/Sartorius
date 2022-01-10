from argparse import ArgumentParser
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from typing import Optional
from cucim.skimage.color import gray2rgb

import cupy as cp

from monai.networks.nets.unetr import UNETR

from bigkingxxl.evaluator.evalutator import iou_map, label_instances
from bigkingxxl.loss.BCEDiceLoss import DiceBCELoss
from bigkingxxl.loss.IoULoss import IoULoss
import torchvision

from bigkingxxl.utils.cupy_tensor import cupyToTensor, tensorToCupy

REAL_LABEL = 1
FAKE_LABEL = 0

class UnetrTrainer(LightningModule):
    def __init__(self,
        data_module: LightningDataModule,
        lr: float,
        threshold: float,
        channels: int = 3,
        height: int = 720,
        width: int = 720,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["data_module", "width", "height", channels])
        data_shape = (channels, width, height)
        self.width = width
        self.height = height
        self.generator = UNETR(in_channels=1, out_channels=channels, img_size=(self.height, self.width), spatial_dims=2)
        self.loss = torch.nn.CrossEntropyLoss()
        self.data_module = data_module

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("UNETR")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--threshold", type=float, default=0.3)
        return parent_parser

    def adverserial_loss(self, y, y_hat):
        return self.loss(y, y_hat)

    def forward(self, X):
        return self.generator(X)

    def binarize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        threshold = self.hparams.threshold
        return (mask > threshold).type_as(mask)

    def training_step(self, batch, batch_idx):
        imgs, real_masks = batch
        generated_masks = self.generator(imgs)
        binary_generated_mask = generated_masks # self.binarize_mask(generated_masks)
        loss = self.adverserial_loss(real_masks, binary_generated_mask)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr)
        lr_scheduler = None # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        if lr_scheduler is None:
            return optimizer
        return { "monitor": "val_iou", "lr_scheduler": lr_scheduler, "optimizer": optimizer }
    
    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        imgs, real_masks = batch
        generated_masks = self.generator(imgs)
        binary_generated_masks = tensorToCupy(self.binarize_mask(generated_masks))
        generated_instances = label_instances(binary_generated_masks)
        combined_instance_masks = self.combine_instances(generated_instances, generated_masks)
        iou_score = iou_map(list(combined_instance_masks), tensorToCupy(real_masks))
        self.log_dict({'val_iou': iou_score.tolist()}, prog_bar=True)
    
    def combine_instances(self, instance_masks: cp.ndarray, logits: torch.Tensor) -> cp.ndarray:
        """Combines a batch of instances of different cell-types into one layer of instances.
        Removes overlaps by using the highest logit found in the logits tensor.

        Args:
            instance_masks (torch.Tensor): Tensor of size (batch size, number of cell-types, width, height) with numbered instances in each layer.
            logits (torch.Tensor): Tensor of size (batch size, number of cell-types, width, height) with logits.

        Returns:
            torch.Tensor: Tensor of size (batch size, width, height) with numbered instances.
        """
        combined_instances = cp.zeros((instance_masks.shape[0], instance_masks.shape[2], instance_masks.shape[3]))
        for batch_index, batch in enumerate(instance_masks):
            max_layers = [0] + [int(batch[layer, :, :].max()) for layer in range(batch.shape[0])][:-1]    
            for index in range(1, len(max_layers)):
                max_layers[index] += max_layers[index - 1]
                instance_masks[batch_index][index] += max_layers[index]

        for batch_index, batch in enumerate(logits):
            maximum = tensorToCupy(torch.argmax(batch, dim=0))
            combined_instances[batch_index] = cp.take_along_axis(instance_masks[batch_index], cp.array([maximum]), axis=0)
        return combined_instances
    
    def on_epoch_end(self) -> None:
        X, y = self.data_module.train_dataloader().dataset[0]
        X = X.reshape(1, 1, self.height, self.width).to(device=next(self.generator.parameters()).device)
        sample_imgs = self.generator(X)
        grid = torchvision.utils.make_grid([cupyToTensor(gray2rgb(tensorToCupy(X.reshape(-1, self.height, self.width)[0]))).permute(2, 0, 1).cpu(), sample_imgs.reshape(-1, self.height, self.width).cpu(), y.cpu()], nrow=3)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
