# BigKingXXL - Sartorius

## Get Started

This section explains how to get started with setting up this repository for developing.

### Download Datasets

To download the challenge dataset simply run `./get-data.sh`. It will download the official kaggle dataset and annotations converted to the coco format provided by [Slawek Biel](https://www.kaggle.com/slawekbiel/sartorius-cell-instance-segmentation-coco).

The data is downloaded into the dataset folder.

### Preprocess Datasets

To preprocess the data run `./preprocess-data.py`. It will transform the run-length encoded masks into binary masks which are used for the `UNET-R GAN`. It also creates TIF masks in `dataset/tif/...` for use with CELLPOSE.

## What we did
During the competition we tried many different approaches. Some of them wre motivated by our own ideas while others came from keeping up to date with the discussion thread on kaggle.

### Detectron2 GAN

### UNET-R GAN (MONAI)

To run the UNET-R approach run `python3 main.py`. Make sure all dependencies are installed. It uses `pytorch-lightning`  as a framework to simplify the code structure and saves results in the folder `lightning-logs`. Results are stored and viewable using `tensoreboard`.

#### Code for the UNET-R GAN

The code of the UNET-R GAN can be found in the `bigkingxxl` directory.

### mmedetection Cascade-RCNN

### mmdetection SWIN Transformer

When looking at different state of the art approaches for image segementation we found segmentation models using a sliding window (SWIN) transformer as backbone achieving top scores. This is why we also tried this approach using mmdetection.
The configuration file for this can be found at `mmdnet_config/sartorius_swin.py`. Our top score achieved this way was:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.142
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.349
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.083
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.140
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.052
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.197
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.197
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.197
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.196
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.074
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.000
```
Since we were not able to run submit models in kaggle without errors we can not provide a score on the test set.
### Detetcron2 without pretraining

### Detectron2 with pretraining

### Detectron2 ensemble classifier

### Detectron2 optimal threshold finding

### Detectron2 minimal cell size

### Cellpose