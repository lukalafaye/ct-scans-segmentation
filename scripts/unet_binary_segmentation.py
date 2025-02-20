import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "scripts")))

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from utils import load_dataset
import pandas as pd

data_dir = "data/"
# output_path = "data/submissions/mask_rcnn.csv"

y_train = pd.read_csv(f"{data_dir}\y_train.csv", index_col=0).T
x_train = load_dataset(f"{data_dir}/train-images")
x_test = load_dataset(f"{data_dir}/test-images")

model = maskrcnn_resnet50_fpn(pretrained=True)
model.train(x_train, y_train)
model.predict(x_test)

# labels.T.to_csv(output_csv_path)