# BigKingXXL - Sartorius

## Get Started

This section explains how to get started with setting up this repository for developing.

### Download Datasets

To download the challenge dataset simply run `./get-data.sh`. It will download the official kaggle dataset and annotations converted to the coco format provided by [Slawek Biel](https://www.kaggle.com/slawekbiel/sartorius-cell-instance-segmentation-coco).

The data is downloaded into the dataset folder.

### Preprocess Datasets

To preprocess the data run `./preprocess-data.py`. It will transform the run-length encoded masks into binary masks which are used for the `UNET-R GAN`. It also creates TIF masks in `dataset/tif/...` for use with CELLPOSE.

## What we did

### Detectron2 GAN

### UNET-R GAN (MONAI)

To run the UNET-R approach run `python3 main.py`. Make sure all dependencies are installed. It uses `pytorch-lightning`  as a framework to simplify the code structure and saves results in the folder `lightning-logs`. Results are stored and viewable using `tensoreboard`.

#### Code for the UNET-R GAN

The code of the UNET-R GAN can be found in the `bigkingxxl` directory.

### mmedetection Cascade-RCNN

### mmdetection SWIN Transformer

### Detetcron2 without pretraining

### Detectron2 with pretraining

### Detectron2 ensemble classifier

### Detectron2 optimal threshold finding

### Detectron2 minimal cell size

### Cellpose