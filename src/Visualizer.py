import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader


class Visualizer():

    def __init__(self, trainDataLoader):
        imgs, masks = next(iter(trainDataLoader))
        self.images = imgs
        self.masks = masks

    def VisualizeEightImages(self, type, rowCount = 4):
        images = self.images[:8]
        masks = self.masks[:8]

        colCount = 4
        fig, ax = plt.subplots(figsize=(16, 16), nrows = 4, ncols= colCount)
        for r in range(rowCount):
            img1 = self.images[2 * r].numpy()
            img1 = np.transpose(img1, (1,2,0))
            msk1 = self.masks[2 * r].numpy()

            img2 = self.images[2 * r + 1].numpy()
            img2 = np.transpose(img2, (1,2,0))
            msk2 = self.masks[2 * r + 1].numpy()

            ax[r, 0].imshow(img1)
            ax[r, 0].set_title('Image: ' + str(r))
            ax[r,1].imshow(msk1)
            ax[r,1].set_title('Mask: ' + str(r))

            ax[r, 2].imshow(img2)
            ax[r, 2].set_title('Image: ' + str(r))
            ax[r,3].imshow(msk2)
            ax[r,3].set_title('Mask: ' + str(r))

        title = 'Visualization of eight images from ' + type
        fig.suptitle(title, fontsize=20)
        fig.savefig(title)


def VisualizePrediction(image, predicted_mask, ground_truth, batchImageName):
    num_images = image.shape[0]

    for index in range(num_images):
        img = image[index].numpy()
        img = np.transpose(img, (1,2,0))
        pred = predicted_mask[index].numpy()
        gt = ground_truth[index].numpy()

        fig, ax = plt.subplots(figsize=(4, 8), nrows=1, ncols=3)
        ax[0].set_title('Statellite Image')
        ax[0].imshow(img)


        ax[1].set_title('Model prediction')
        ax[1].imshow(pred)

        ax[2].set_title('Ground Truth')
        ax[2].imshow(gt)

        title = 'Model Prediction'
        fig.suptitle(title, fontsize=20)
        fig.savefig(batchImageName + str(index))