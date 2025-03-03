#!/bin/bash

# Install dependencies


# fetch exact torch command using pytorch install website
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install nnunetv2 wandb
pip install triton opencv-python tensorflow

# Download images
wget https://challengedata.ens.fr/media/public/train-images.zip
wget https://challengedata.ens.fr/media/public/test-images.zip
wget https://challengedata.ens.fr/media/public/label_Hnl61pT.csv -O y_train.csv
wget https://challengedata.ens.fr/media/public/annotated_labels.json

# Unzip images
unzip -n train-images.zip
unzip -n test-images.zip

rm -rf *.zip
