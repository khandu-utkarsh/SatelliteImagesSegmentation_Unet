import os
import sys
import time
'''
def install(package):
    sys.subprocess.check_call([sys.executable, "-m","pip","install",package])
'''
import SegmentationDataset as sdata
from torch.utils.data import DataLoader
import SegmentationModel as sm
from segmentation_models_pytorch import losses
from trainloop import training_loop, segmentation_test_loop, class_report
import torch
import torch.utils.data as data_utils

workspaceRoot = os.getcwd()

indices = torch.arange(8)

trainDataset = sdata.Landcover_ai_Dataset(workspaceRoot)
testDataset = sdata.Landcover_ai_Dataset(workspaceRoot, mode = "test")
validDataset = sdata.Landcover_ai_Dataset(workspaceRoot, mode = "val")

train16 = data_utils.Subset(trainDataset, indices)
val16 = data_utils.Subset(validDataset, indices)
test16 = data_utils.Subset(testDataset, indices)

# Hyperparameters
batch_size = 8
epochs = 3
lr = 5e-5
loss_fn = losses.JaccardLoss(mode = "multiclass",classes = [0,1,2,3,4])
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
