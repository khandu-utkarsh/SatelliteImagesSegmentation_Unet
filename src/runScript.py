import os
import SegmentationDataset as sdata
from torch.utils.data import DataLoader
import SegmentationModel as sm
from segmentation_models_pytorch import losses
from trainloop import training_loop
import torch

workspaceRoot = os.getcwd()

trainDataset = sdata.Landcover_ai_Dataset(workspaceRoot)
testDataset = sdata.Landcover_ai_Dataset(workspaceRoot, mode = "test")
validDataset = sdata.Landcover_ai_Dataset(workspaceRoot, mode = "val")

# Hyperparameters
batch_size = 8
epochs = 1
lr = 5e-5
loss_fn = losses.JaccardLoss(mode = "multiclass",classes = [0,1,2,3,4])

train_dloader = DataLoader(trainDataset,batch_size = batch_size)
test_dloader = DataLoader(testDataset, batch_size = batch_size)
val_dloader = DataLoader(validDataset, batch_size=batch_size,)

segmentationModel =  sm.SegmentationModel()
segmentationModel.InitializeModel()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = segmentationModel.model.to(device)
trainOut = training_loop(model, train_dloader, val_dloader, epochs, lr, loss_fn)