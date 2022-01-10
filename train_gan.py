from bigkingxxl.dataloader.dataloader import SartoriusDataLoader
from monai.networks.nets.unetr import UNETR
from bigkingxxl.discriminator.discriminator import Discriminator
from bigkingxxl.trainer.gan_trainer import GanTrainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser

def main():
    # define generator and discriminator
    parser = ArgumentParser()
    GanTrainer.add_model_specific_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    # Define the generator
    generator = UNETR(in_channels=1, out_channels=3, img_size=(512, 704), spatial_dims=2)
    # Define the discriminator
    discriminator = Discriminator((4, 512, 704))
    # We want to save the weights of every epoch
    checkpoint_callback = ModelCheckpoint(every_n_epochs = 1, auto_insert_metric_name=True, save_top_k=-1)
    # Define which dataloader we want to use
    datamodule = SartoriusDataLoader()
    # Instantiate the GAN Trainer module
    gan = GanTrainer(
        generator,
        discriminator,
        datamodule,
        **dict_args
    )
    # Define how to train the network
    trainer = Trainer(gpus=1, gradient_clip_val=0.01, check_val_every_n_epoch=1, callbacks=[checkpoint_callback], log_every_n_steps=10)
    # Start training
    trainer.fit(gan, datamodule=datamodule)

if __name__ == '__main__':
    main()
