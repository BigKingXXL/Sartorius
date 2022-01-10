#!/bin/bash
# This script download the competition data aswell as a coco version of it and puts everything into the right directories.
export PWD=$(pwd)
set -e
trap "cd $PWD; exit" INT

if ! command -v unzip &> /dev/null
then
    echo "unzip is not installed"
    echo "please make sure the command unzip works"
    exit
fi
if ! command -v kaggle &> /dev/null
then
    echo "kaggle is not installed"
    echo "please make sure the command kaggle works"
    exit
fi
echo "download compressed kaggle challenge dataset"
kaggle competitions download -c sartorius-cell-instance-segmentation -p dataset
echo "decompressing compressed kaggle challenge dataset"
cd dataset && unzip sartorius-cell-instance-segmentation.zip; cd ..
echo "removing compressed kaggle challenge dataset"
rm dataset/sartorius-cell-instance-segmentation.zip
echo "download compressed kaggle challenge dataset annotations in coco format"
kaggle datasets download -d slawekbiel/sartorius-cell-instance-segmentation-coco -p dataset
echo "decompressing compressed kaggle challenge dataset annotations in coco format"
cd dataset && unzip sartorius-cell-instance-segmentation-coco.zip; cd ..
echo "removing compressed kaggle challenge dataset annotations in coco format"
rm dataset/sartorius-cell-instance-segmentation-coco.zip
echo "success"