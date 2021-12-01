from monai.networks.nets.unetr import UNETR
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from bigkingxxl.discriminator.discriminator import Discriminator
from bigkingxxl.evaluator.evalutator import label_instances
from bigkingxxl.loss.IoULoss import IoULoss
from bigkingxxl.trainer.gan_trainer import GanTrainer
from bigkingxxl.dataset.dataset import SartoriusDataset
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import DataLoader
import logging
import matplotlib.pyplot as plt

DEVICE = 'cuda'
SAVE = 'models/2021_11_29_22_39_19_014861/epoch_6_'

def main(device: str = DEVICE):
    # define generator and discriminator
    #generator = UNETR(in_channels=1, out_channels=1, img_size=(520, 704), spatial_dims=2)
    generator = UNETR(in_channels=1, out_channels=3, img_size=(512, 704), spatial_dims=2).to(device)
    discriminator = Discriminator((4, 512, 704)).to(device)
    
    generator.load_state_dict(torch.load(SAVE + 'generator.pth'))

    train_dataset = SartoriusDataset(dataset_path = './dataset', mode = 'train')
    train_dataloader = DataLoader(train_dataset)
    # define test data
    test_dataset = SartoriusDataset(dataset_path = './dataset', mode = 'test')
    test_dataloader = DataLoader(test_dataset)

    for inputImage, maskImage in test_dataloader:
        inputImage = inputImage[:, :512, :].reshape(-1, 1, 512, 704).float().to(device)
        maskImage = maskImage[:, :, :512,:].reshape(-1, 3, 512, 704).float().to(device)
        predictedMask = generator(inputImage).reshape(3, 512, 704)
        plt.imshow(predictedMask.permute(1, 2, 0))
        plt.show()
        label_instances(predictedMask)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
