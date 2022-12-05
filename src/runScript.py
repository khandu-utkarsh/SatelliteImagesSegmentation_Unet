import os

import numpy as np

import SegmentationDataset as sdata
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import SegmentationModel as sm
from segmentation_models_pytorch import losses
from trainloop import training_loop
import torch
import Visualizer as v
workspaceRoot = os.getcwd()

#Restore this back when done development
trainDataset = sdata.Landcover_ai_Dataset(workspaceRoot)
trainDataset = data_utils.Subset(trainDataset, torch.arange(16))
testDataset = sdata.Landcover_ai_Dataset(workspaceRoot, mode = "test")
validDataset = sdata.Landcover_ai_Dataset(workspaceRoot, mode = "val")
validDataset = data_utils.Subset(validDataset, torch.arange(16))
# Hyperparameters
batch_size = 8
epochs = 1
lr = 5e-5
loss_fn = losses.JaccardLoss(mode = "multiclass",classes = [0,1,2,3,4])

#Loading the datasets and visualization part
train_dloader = DataLoader(trainDataset,batch_size = batch_size)
visualizer1 = v.Visualizer(train_dloader)
visualizer1.VisualizeEightImages('training set')
visualizer1 = None

test_dloader = DataLoader(testDataset, batch_size = batch_size)
visualizer2 = v.Visualizer(test_dloader)
visualizer2.VisualizeEightImages('testing set')
visualizer2 = None

val_dloader = DataLoader(validDataset, batch_size=batch_size,)
visualizer2 = v.Visualizer(val_dloader)
visualizer2.VisualizeEightImages('validation set')
visualizer2 = None

segmentationModel =  sm.SegmentationModel()
segmentationModel.InitializeModel()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = segmentationModel.model.to(device)
trainOut = training_loop(model, train_dloader, val_dloader, epochs, lr, loss_fn)