import torch.nn as nn
import math

from typing import Tuple

# angelehnt an https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/361f8b00d671b66db752e66493b630be8bc7d67b/models/networks.py#L538
class Discriminator(nn.Module):
    """
    A basic CNN to be used as a descriminator in a GAN.
    It adapts the model from:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/361f8b00d671b66db752e66493b630be8bc7d67b/models/networks.py#L538
    """
    def __init__(self, sizes = (4, 512, 704)) -> None:
        super(Discriminator, self).__init__()
        self.net = self.build_net(sizes)
    
    def build_net(self, sizes: Tuple) -> nn.Sequential:
        layers = []
        currentDimensions = sizes[0]
        currentWidth = sizes[1]
        currentHeight = sizes[2]
        
        while currentHeight >= 32 and currentWidth >= 32:
            layers += [
                nn.Conv2d(currentDimensions, 2*currentDimensions, stride=2, padding=1, kernel_size=4),
                nn.LeakyReLU(0.2, True)
            ]

            # Update current dimensions, width and height variables
            currentDimensions *= 2
            currentHeight = max(1, math.floor(currentHeight / 2))
            currentWidth = max(1, math.floor(currentWidth / 2))
        
        # Reduce to one dimension and apply sigmoid
        # layers.append(nn.Conv2d(currentDimensions, 1, kernel_size=1, padding=0, stride=1))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(currentDimensions * currentHeight * currentWidth, 1))
        # layers.append(nn.Sigmoid())
        
        # Map list to a 
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)