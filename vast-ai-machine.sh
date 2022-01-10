#!/bin/bash
set -e
apt update
apt install git unzip gcc nano ffmpeg libsm6 libxext6 g++ -y
pip install pytorch-lightning pandas scipy scikit-learn scikit-image tiffile kaggle  opencv-python
mkdir ~/.kaggle -p
nano ~/.kaggle/kaggle.json
git clone https://max-3l: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@github.com/BigKingXXL/Sartorius.git
cd Sartorius
git checkout pytorch-lightning
/bin/bash ./get-data.sh
python3 preprocess-dataset.py
pip install openmim
mim install mmdet
git clone https://github.com/open-mmlab/mmdetection.git
pip install instaboostfast git+https://github.com/cocodataset/panopticapi.git git+https://github.com/lvis-dataset/lvis-api.git
pip install albumentations>=0.3.2 --no-binary imgaug,albumentations
