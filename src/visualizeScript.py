import os
import argparse
from torch.utils.data import DataLoader
import SegmentationDataset as sdata
import SegmentationModel as sm
from Visualizer import VisualizePrediction
import torch.nn.functional as F
import torch
import torch.utils.data as data_utils

class TestClass:
    def __init__(self, segmentation_model, testLoader, modelRelativePathFromRoot):
        self.test_loader = testLoader

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.visualizations_dir = os.path.join(segmentation_model.workspace_root_dir, "visualizations")         #Vizualization Directory
        self.model = segmentation_model.model
        self.modelRelativePathFromRoot = modelRelativePathFromRoot

        modelPath = os.path.join(segmentation_model.workspace_root_dir, self.modelRelativePathFromRoot)
        print("Loading the saved model from: ", modelPath)
        self.model.load_state_dict(torch.load(modelPath, map_location=torch.device(self.device)))

    # Test loop
    def segmentation_predict_loop(self):
        model = self.model.to(self.device)
        model.eval()

        for batch_index, (X,y) in enumerate(self.test_loader):
            X = X.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                logits = F.softmax(model(X), dim =1)
                probs, preds = torch.max(logits, dim = 1)

            imageFileName = os.path.join(self.visualizations_dir, "Predictions_" + str(batch_index))
            VisualizePrediction(X, preds, y, imageFileName)
        return

def VisualizePredictions(workspaceRoot,relModelPathFromRoot, batch_size = 4):
    testDataset = sdata.Landcover_ai_Dataset(workspaceRoot, mode = "test")
    #testDataset = data_utils.Subset(testDataset, torch.arange(8))  #Have to remove this
    test_dloader = DataLoader(testDataset, batch_size = batch_size)
    segmentationModel =  sm.SegmentationModel(workspaceRoot)
    tester = TestClass(segmentationModel, test_dloader, relModelPathFromRoot)
    print('Inference Starts here: ')
    tester.segmentation_predict_loop()
    return True


parser = argparse.ArgumentParser(description='Provide arguments if you want to use existing trained model.')
parser.add_argument('--model-name', dest = 'modelName', default = None, type = str, help='Provide name of the model from workspaceRoot', required = False)

workspaceRoot = os.getcwd()
#workspaceRoot = '/Users/utkarsh/GitHubCodeRepositories/SatelliteImagesSegmentation_Unet'
batch_size = 4

args = parser.parse_args()
print('Args: ',args)
modelRelativePath = args.modelName
print(workspaceRoot, batch_size,modelRelativePath)
VisualizePredictions(workspaceRoot,modelRelativePath, batch_size)