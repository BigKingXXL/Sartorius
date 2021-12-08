from bigkingxxl.dataloader.dataloader import SartoriusDataLoader
from monai.networks.nets.unetr import UNETR
from bigkingxxl.discriminator.discriminator import Discriminator
from bigkingxxl.trainer.gan_trainer import GanTrainer
from bigkingxxl.dataset.dataset import SartoriusDataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from argparse import ArgumentParser

def main():
    # define generator and discriminator
    parser = ArgumentParser()
    GanTrainer.add_model_specific_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    generator = UNETR(in_channels=1, out_channels=3, img_size=(512, 704), spatial_dims=2)
    discriminator = Discriminator((4, 512, 704))
    gan = GanTrainer(
        generator,
        discriminator,
        **dict_args
    )
    datamodule = SartoriusDataLoader()
    trainer = Trainer(gpus=0, fast_dev_run=True)
    trainer.fit(gan, datamodule=datamodule)

if __name__ == '__main__':
    main()
