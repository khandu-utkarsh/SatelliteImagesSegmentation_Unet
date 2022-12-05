import os
import numpy as np

import SegmentationDataset as sdata
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import SegmentationModel as sm
from segmentation_models_pytorch import losses
from trainloop import training_loop, segmentation_test_loop, class_report
import torch

import Visualizer as v
workspaceRoot = os.getcwd()

import torch.utils.data as data_utils

workspaceRoot = os.getcwd()

indices = torch.arange(8)

trainDataset = sdata.Landcover_ai_Dataset(workspaceRoot)
trainDataset = data_utils.Subset(trainDataset, torch.arange(16))
testDataset = sdata.Landcover_ai_Dataset(workspaceRoot, mode = "test")
validDataset = sdata.Landcover_ai_Dataset(workspaceRoot, mode = "val")
validDataset = data_utils.Subset(validDataset, torch.arange(16))

train16 = data_utils.Subset(trainDataset, indices)
val16 = data_utils.Subset(validDataset, indices)
test16 = data_utils.Subset(testDataset, indices)

# Hyperparameters
batch_size = 8
epochs = 3
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


classes = [0,1,2,3,4]
train_dloader = DataLoader(train16,batch_size = batch_size)
test_dloader = DataLoader(test16, batch_size = batch_size)
val_dloader = DataLoader(val16, batch_size=batch_size,)

segmentationModel =  sm.SegmentationModel()
segmentationModel.InitializeModel()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)
model = segmentationModel.model.to(device)
trainOut = training_loop(model, train_dloader, val_dloader, epochs, lr, loss_fn, save = True, model_title = "Segtry") 
statsc, acc, mlacc, jaccard, class_probs = segmentation_test_loop(model, test_dloader)
print("Statscores: ",statsc, statsc.shape)
class_report(classes, statsc, acc, mlacc, jaccard, class_probs)
