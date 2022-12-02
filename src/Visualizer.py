import os
import glob
import cv2
#import matplotlib.pyplot as plt
#import matplotlib.colors
#import seaborn as sns

import numpy as np


class Visualizer():
    def __init__(self, working_dir):
        self.images_dir = os.path.join(os.getcwd(), "images")
        self.masks_dir = os.path.join(os.getcwd(), "masks")
        self.output_dir = os.path.join(os.getcwd(), "output")  #   This is the directory where imaegs are located

    def VisualizeBlock(self, num_samples = 4, seed = 123):
        data_list = list(glob.glob(os.path.join(self.output_dir, "*.jpg")))
        np.random.seed(seed)
        indices = np.random.randint(low=0, high=len(data_list), size=num_samples)
        for i, idx in enumerate(indices):
            img = cv2.imread(data_list[idx]) / 255
            mask_pt = data_list[indices[i]].split(".jpg")[0] + "_m.png"
            mask = cv2.imread(mask_pt)[:, :, 1]

workingDir = "/Users/utkarsh/GitHubCodeRepositories/SatelliteImagesSegmentation_Unet/landcover.ai.v1"
vizObj = Visualizer(workingDir)

vizObj.VisualizeBlock()