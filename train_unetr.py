from bigkingxxl.dataloader.dataloader import SartoriusDataLoader, SartoriusDataLoaderPadded, SartoriusDataLoaderSquare, SartoriusDataLoaderUnscaled
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import ArgumentParser

from bigkingxxl.trainer.unetr_trainer import UnetrTrainer

def main():
    parser = ArgumentParser()
    UnetrTrainer.add_model_specific_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    # We want to save every epoch
    checkpoint_callback = ModelCheckpoint(every_n_epochs = 5, auto_insert_metric_name=True, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # Use a padding dataloader which we defined to be suitable for working with the UNET-R
    datamodule = SartoriusDataLoaderPadded(height=528, width=704)
    # Instantiate the UNET-R trainer module
    unetr = UnetrTrainer(
        datamodule,
        **dict_args,
        height=528,
        width=704
    )
    # Define training parameters
    trainer = Trainer(gpus=1, gradient_clip_val=0.01, check_val_every_n_epoch=1, callbacks=[checkpoint_callback, lr_monitor], log_every_n_steps=10)
    # Start the training
    trainer.fit(unetr, datamodule=datamodule)

if __name__ == '__main__':
    main()
