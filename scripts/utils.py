import pandas as pd
from pathlib import Path
import cv2
import numpy as np
from skimage.filters import rank
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.morphology import disk
from tqdm import tqdm
from scipy import ndimage as ndi

def load_dataset(dataset_dir):
    dataset_list = []
    # Note: It's very important to load the images in the correct numerical order!
    for image_file in list(sorted(Path(dataset_dir).glob("*.png"), key=lambda filename: int(filename.name.rstrip(".png")))):
        dataset_list.append(cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE))
    return np.stack(dataset_list, axis=0)

def compute_baseline_one_sample(data_slice):
    edges = sobel(data_slice)
    denoised = rank.median(data_slice, disk(2))
    markers = rank.gradient(denoised, disk(5)) < 20
    markers = ndi.label(markers)[0]
    label_predicted = watershed(edges, markers=markers, compactness=0.0001)
    return label_predicted

def compute_baseline(dataset: np.array):
    labels_predicted_list = []
    for data_index in tqdm(range(len(dataset))):
        data_slice = dataset[data_index]
        label_predicted = compute_baseline_one_sample(data_slice)
        labels_predicted_list.append(label_predicted)
    return pd.DataFrame(np.stack(labels_predicted_list, axis=0).reshape((len(labels_predicted_list), -1)))

def get_label_matrix_from_csv(label_csv_path):
    return pd.read_csv(label_csv_path, index_col=0, header=0).T.values.reshape((-1, 256, 256))

def save_predictions_to_csv(predictions, output_csv_path):
    ''' Predictions is a list of 2D predictions, i.e. a 3D numpy array of shape (500, 256, 256) for the test prediction / submission
    '''
    pd.DataFrame(predictions.reshape((predictions.shape[0], -1))).T.to_csv(output_csv_path)