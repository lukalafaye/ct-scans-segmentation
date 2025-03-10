#!/bin/bash

# Install dependencies

# fetch exact torch command using pytorch install website
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt

# Download images
wget https://challengedata.ens.fr/media/public/train-images.zip
wget https://challengedata.ens.fr/media/public/test-images.zip
wget https://challengedata.ens.fr/media/public/label_Hnl61pT.csv -O y_train.csv
wget https://challengedata.ens.fr/media/public/annotated_labels.json

# Unzip images
unzip -n train-images.zip
unzip -n test-images.zip

rm -rf *.zip

mkdir -p ../data
mv y_train.csv annotated_labels.json train-images test-images ../data/
