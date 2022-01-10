# BigKingXXL - Sartorius

## Get Started

This section explains how to get started with setting up this repository for developing.

### Download Datasets

To download the challenge dataset simply run `./get-data.sh`. It will download the official kaggle dataset and annotations converted to the coco format provided by [Slawek Biel](https://www.kaggle.com/slawekbiel/sartorius-cell-instance-segmentation-coco).

The data is downloaded into the dataset folder.

### Preprocess Datasets

To preprocess the data run `./preprocess-data.py`. It will transform the run-length encoded masks into binary masks which are used for the `UNET-R GAN`. It also creates TIF masks in `dataset/tif/...` for use with CELLPOSE.

## What we did

In the next

### Detectron2 GAN

Our first idea was to use the predictions of a Detectron2 instance in a GAN to use the unlabled data to train. But as we used bitmasks as inputs and detectron2's bitmask output has no gradient we did not follow this approach, as it would involve rewriting a big part of detectrons code.

### UNET-R GAN (MONAI)

To run the UNET-R approach run `python3 main.py`. Make sure all dependencies are installed. It uses `pytorch-lightning`  as a framework to simplify the code structure and saves results in the folder `lightning-logs`. Results are stored and viewable using `tensoreboard`.

#### Code for the UNET-R GAN

The code of the UNET-R GAN can be found in the `bigkingxxl` directory.

### mmedetection Cascade-RCNN

### mmdetection SWIN Transformer

### Detectron2

We tried different ways to use and train detectron2. With and without pretraining, different architectures and three detectron instances as an ensemble classifier. We also computed the minimal cell size to filter out too small predictions and used a greedy algorithm to optimize the thresholds. The notebooks for training and inference can be found in the `detectron/` directory.

### Cellpose

We converted our computed masks to tiff files and used them to train a CellPose model that is pretrained on nuclei images. More explanations can be found in the notebook `Cellpose.ipynb`.
