import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import cv2
import random

class Visualizer():

    def __init__(self, images, masks, index, wp):
        #imgs, masks = next(iter(trainDataLoader))
        imgs, masks = images, masks
        self.images = imgs
        self.masks = masks
        self.workspaceRootDir = wp
        self.index = index

    def VisualizeEightImages(self, type, rowCount = 4):
        label_to_name_dict = {0 : 'Background', 1: 'Building', 2: 'Woodland', 3: 'Water', 4: 'Road'}
        name_to_color_code = {'Background' : '#566573', #Grey
                                'Building': '#884EA0' , #Meganta
                                'Woodland' : '#28B463', #Green
                                'Water ' : '#2874A6', #Water
                                'Road' : '#FFFF00'} #Yellow

        #cols = name_to_color_code.values()
        cols = ['#566573', '#884EA0', '#28B463','#2874A6','#FFFF00']
        current_color_map = matplotlib.colors.ListedColormap(cols)

        
        #current_color_map = matplotlib.colors.ListedColormap(cols)

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
            ax[r, 0].set_title('Image: ' + str(2 * r + 1))
            ax[r,1].imshow(msk1, cmap=current_color_map, vmin=0, vmax=4)
            ax[r,1].set_title('Mask: ' + str(2 * r + 1))            
            ax[r, 2].imshow(img2)
            ax[r, 2].set_title('Image: ' + str(2 * r + 2))
            ax[r,3].imshow(msk2, cmap=current_color_map, vmin=0, vmax=4)
            ax[r,3].set_title('Mask: ' + str(2 * r + 2))

        title = 'Visualization of eight images from ' + type
        fig.suptitle(title, fontsize=20)

        fileName = "Visualization_of_eight_images_from_" + type + '_batch_index_' + str(self.index)
        visualizationPath = os.path.join(self.workspaceRootDir, "visualizations")
        fileFullPath = os.path.join(visualizationPath, fileName)
        fig.savefig(fileFullPath)
        plt.close(fig)


def VisualizePrediction(image, predicted_mask, ground_truth, batchImageName):
    label_to_name_dict = {0 : 'Background', 1: 'Building', 2: 'Woodland', 3: 'Water', 4: 'Road'}
    name_to_color_code = {'Background' : '#566573',
                          'Building': '#884EA0' ,
                          'Woodland' : '#28B463',
                          'Water ' : '#2874A6',
                          'Road' : '#FFFF00'}

    #cols = name_to_color_code.values()
    cols = ['#566573', '#884EA0', '#28B463','#2874A6','#FFFF00']
    current_color_map = matplotlib.colors.ListedColormap(cols)

    num_images = image.shape[0]

    for index in range(num_images):
        img = image[index].cpu().numpy()
        img = np.transpose(img, (1,2,0))
        pred = predicted_mask[index].cpu().numpy()
        gt = ground_truth[index].cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 4), nrows=1, ncols=3)
        ax[0].set_title('Statellite Image')
        ax[0].imshow(img)


        ax[1].set_title('Model prediction')
        ax[1].imshow(pred, cmap=current_color_map, vmin=0, vmax=4)
        
        ax[2].set_title('Ground Truth')
        ax[2].imshow(gt, cmap=current_color_map, vmin=0, vmax=4)

        title = 'Model Prediction'
        fig.suptitle(title, fontsize=20)
        fig.savefig(batchImageName +'_' + str(index))
        plt.close(fig)

def VisualizeTransforms(transforms, transforms_names,visPath):
    NUM_SAMPLE = random.randint(1,10)
    trainpath_list = list(glob.glob(os.path.join(os.getcwd(),"datasets_images", "*.jpg")))
    img = cv2.imread(trainpath_list[NUM_SAMPLE])

    fig, ax = plt.subplots(figsize = (10,10), nrows = 2, ncols = 3)

    ax[0,0].imshow(img)
    ax[0,0].axis("off")
    ax[0,0].set_title("True image")
    count = 0

    for i in range(2):
        for j in range(3):
            if i+j == 0:
                ax[i,j].imshow(img)
                ax[i,j].axis("off")
                ax[i,j].set_title("True image")
            else:
                transformed_img = transforms[count](image = img)["image"]
                ax[i,j].imshow(transformed_img)
                ax[i,j].axis("off")
                ax[i,j].set_title(transforms_names[count])
                count+=1
    plt.suptitle("Data augmentation",fontsize=20)
    plt.tight_layout(pad = 1)
    savefigpath = os.path.join(visPath,"Sample_Augmentations_{}.png".format(NUM_SAMPLE))
    plt.savefig(savefigpath)
    plt.close(fig)

def PlotTrainingLossesGraph(counter_epochs, train_loss_list, val_loss_list, saveFigPath):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(range(1, counter_epochs + 1), train_loss_list, label='Train Loss', linewidth=2.5)
    ax.plot(range(1, counter_epochs + 1), val_loss_list, label='Val Loss', linewidth=2.5)
    ax.set_title("Graph between losses and epoch counts", fontsize=15)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_xlabel("Epochs", fontsize=13)
    plt.legend()
    plt.savefig(saveFigPath)
    plt.close(fig)
