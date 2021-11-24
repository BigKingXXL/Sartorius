from monai.networks.nets.unetr import UNETR
from bigkingxxl.discriminator.discriminator import Discriminator
from bigkingxxl.trainer.gan_trainer import GanTrainer
from bigkingxxl.dataset.dataset import SartoriusDataset
from torch.optim import Adam
from torch.nn import BCELoss
import logging
import torch

def main():
    # define generator and discriminator
    #generator = UNETR(in_channels=1, out_channels=1, img_size=(520, 704), spatial_dims=2)
    generator = UNETR(in_channels=1, out_channels=3, img_size=(512, 704), spatial_dims=2)
    discriminator = Discriminator((4, 512, 704))
    # define optimizers
    generator_optim = Adam(generator.parameters(), lr=0.0001)
    discriminator_optim = Adam(discriminator.parameters(), lr=0.0001)
    # define losses
    generator_loss = BCELoss()
    discriminator_loss = BCELoss()
    # define training data
    train_dataset = SartoriusDataset(dataset_path = './dataset', mode = 'train')
    # define test data
    test_dataset = SartoriusDataset(dataset_path = './dataset', mode = 'test')
    # define trainer
    trainer = GanTrainer(
        generator,
        discriminator,
        generator_optim,
        discriminator_optim,
        generator_loss,
        discriminator_loss
    )
    trainer.setTrainingData(train_dataset)
    trainer.setTestData(test_dataset)
    trainer.train(1)
    torch.save(generator.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
