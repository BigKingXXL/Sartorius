from monai.networks.nets.unetr import UNETR
from bigkingxxl.discriminator.discriminator import Discriminator
from bigkingxxl.trainer.gan_trainer import GanTrainer
from bigkingxxl.dataset.dataset import SartoriusDataset
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import DataLoader
import logging
import torch

DEVICE = 'cpu'

def main(device: str = DEVICE):
    # define generator and discriminator
    #generator = UNETR(in_channels=1, out_channels=1, img_size=(520, 704), spatial_dims=2)
    generator = UNETR(in_channels=1, out_channels=3, img_size=(512, 704), spatial_dims=2).to(device)
    discriminator = Discriminator((4, 512, 704)).to(device)
    # define optimizers
    generator_optim = Adam(generator.parameters(), lr=0.0001)
    discriminator_optim = Adam(discriminator.parameters(), lr=0.0001)
    # define losses
    generator_loss = BCELoss()
    discriminator_loss = BCELoss()
    # define training data
    train_dataset = SartoriusDataset(dataset_path = './dataset', mode = 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    # define test data
    test_dataset = SartoriusDataset(dataset_path = './dataset', mode = 'test')
    test_dataloader = DataLoader(test_dataset)
    # define trainer
    trainer = GanTrainer(
        generator,
        discriminator,
        generator_optim,
        discriminator_optim,
        generator_loss,
        discriminator_loss,
        device
    )
    trainer.setTrainingData(train_dataloader)
    trainer.setTestData(test_dataloader)
    trainer.train(1)
    torch.save(generator.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
