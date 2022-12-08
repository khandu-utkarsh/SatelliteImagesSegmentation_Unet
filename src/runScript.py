import os
import argparse

from torch.utils.data import DataLoader

import SegmentationDataset as sdata
import SegmentationModel as sm
import trainloop as segmentation_train_test
import Visualizer as v

#Could remove these once subset functions are being removed
import torch
import torch.utils.data as data_utils

def RunModel(workspaceRoot, batch_size, epochs, lr, useSaveModel = False, modelPath = None):
    trainDataset = sdata.Landcover_ai_Dataset(workspaceRoot)
    trainDataset = data_utils.Subset(trainDataset, torch.arange(16)) #Have to remove this
    train_dloader = DataLoader(trainDataset,batch_size = batch_size)
    visualizer = v.Visualizer(train_dloader,workspaceRoot)
    visualizer.VisualizeEightImages('training set')

    testDataset = sdata.Landcover_ai_Dataset(workspaceRoot, mode = "test")
    testDataset = data_utils.Subset(testDataset, torch.arange(16))  #Have to remove this
    test_dloader = DataLoader(testDataset, batch_size = batch_size)
    visualizer = v.Visualizer(test_dloader,workspaceRoot)
    visualizer.VisualizeEightImages('testing set')


    validDataset = sdata.Landcover_ai_Dataset(workspaceRoot, mode = "val")
    validDataset = data_utils.Subset(validDataset, torch.arange(16))  #Have to remove this
    val_dloader = DataLoader(validDataset, batch_size=batch_size,)
    visualizer = v.Visualizer(val_dloader,workspaceRoot)
    visualizer.VisualizeEightImages('validation set')
    visualizer = None

    #Generate Model
    segmentationModel =  sm.SegmentationModel(workspaceRoot)
    segmentationModel.InitializeModel()

    #Training and Testing Process
    print('Training Started')
    if useSaveModel:
        trainTestObj = segmentation_train_test.TrainTest(segmentationModel, train_dloader, val_dloader, test_dloader, modelPath, True)
    else:
        trainTestObj = segmentation_train_test.TrainTest(segmentationModel, train_dloader, val_dloader, test_dloader)

    trainTestObj.segmentation_training_loop(epochs, lr)
    print('Training Complete')

    #Testing on our dataset
    print('Evaluation Started')
    trainTestObj.segmentation_test_loop()
    print('Evaluation Completed')

    return True

parser = argparse.ArgumentParser(description='Provide arguments if you want to use existing trained model.')
parser.add_argument('--use-existing-model', dest = 'userInputBoolean',default=False, type=bool, required = False)
parser.add_argument('--model-name', dest = 'modelName', default = None, type = str, help='Provide name of the model from ./models', required = False)

args = parser.parse_args()
print(args)

if args.userInputBoolean == True and args.modelName != None:
    useSaveModel = True
    modelPath = args.modelName
    print("Saved model path provided")
else:
    useSaveModel = False
    modelPath = None

workspaceRoot = os.getcwd()
batch_size = 8
epochs = 1
lr = 5e-5

print('Printing all the arguments used by RunModel function:')
print(workspaceRoot, batch_size, epochs, lr, useSaveModel, modelPath)
print('CUDA Availability: ', torch.cuda.is_available())
success = RunModel(workspaceRoot, batch_size, epochs, lr, useSaveModel, modelPath)