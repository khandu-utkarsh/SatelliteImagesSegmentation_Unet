import os
import argparse

from torch.utils.data import DataLoader
import SegmentationDataset as sdata
import SegmentationModel as sm

#Could remove these once subset functions are being removed
import torch
import torch.utils.data as data_utils

class TestClass:
    def __init__(self, segmentation_model, testLoader, modelRelativePathFromRoot):
        self.test_loader = testLoader

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = segmentation_model.model
        self.modelRelativePathFromRoot = modelRelativePathFromRoot
        
        modelPath = os.path.join(segmentation_model.workspace_root_dir, self.modelRelativePathFromRoot)
        print("Loading the saved model from: ", modelPath)
        self.model.load_state_dict(torch.load(modelPath))

    # Test loop
    def segmentation_test_loop(self):
        #Functions for eval
        precision = torchmetrics.Precision(task = "multiclass", num_classes=5, average= None).to(self.device)
        recall = torchmetrics.Recall(task = "multiclass", num_classes=5, average= None).to(self.device)
        F1_score = torchmetrics.F1Score(task = "multiclass", num_classes=5, average= None).to(self.device)
        acc = torchmetrics.Accuracy(num_classes = 5, average = "micro",task="multiclass").to(self.device)
        mlacc = torchmetrics.classification.MulticlassExactMatch(num_classes=5).to(self.device)
        jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes = 5).to(self.device)

        model = self.model.to(self.device)
        model.eval()

        class_probs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        num_samples = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

        for batch_index, (X,y) in enumerate(self.test_loader):
            X = X.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                logits = F.softmax(model(X), dim =1)
                probs, preds = torch.max(logits, dim = 1)

                for label in class_probs.keys():
                    class_probs[label]+= probs[preds == label].flatten().sum()
                    num_samples[label]+= preds[preds == label].flatten().size(dim = 0)

                precision.update(preds, y)
                recall.update(preds, y)
                F1_score.update(preds, y)
                acc.update(preds,y)
                mlacc.update(preds,y)
                jaccard.update(preds, y)

        for label in class_probs.keys():
            class_probs[label] /= num_samples[label]

        print("Model Path: ", modelPath)
        self.class_report(["Background","Building","Woodland","Water","Road"], precision.compute(), recall.compute(), F1_score.compute(), acc.compute(), mlacc.compute(), jaccard.compute(), class_probs)
        return

    def class_report(self, classes, precision, recall, F1score, acc, mlacc, jaccard, class_probs):
        data = [precision.detach(), recall.detach(), F1score.detach(), acc.detach(), mlacc.detach(),jaccard.detach(), class_probs]

        data[0] = data[0].cpu()
        data[1] = data[1].cpu()
        data[2] = data[2].cpu()
        data[3] = data[3].cpu()
        data[4] = data[4].cpu()
        data[5] = data[5].cpu()

        p = data[0].reshape(5,1)
        r = data[1].reshape(5,1)
        f1 = data[2].reshape(5,1)

        table = np.concatenate((p,r,f1), axis = 1)
        table = table.tolist()
        table[0].insert(0,"Background")
        table[1].insert(0,"Building")
        table[2].insert(0,"Woodland")
        table[3].insert(0,"Water")
        table[4].insert(0,"Road")


        print(tabulate(table, headers=["Classes", "Precision", "Recall", "F1 Score"]))

        print("\n")
        print("Classification Accuracy: ", data[3].numpy())
        print("Exact Match Ratio: ", data[4].numpy())
        print("Jaccard Index: ", data[5].numpy())
        print("\n")

        print('Class Probabilities:')
        for idx in class_probs.keys():
            print(f"{classes[idx]}:{class_probs[idx].cpu():.3f}")
        return



def PrintPredictions(workspaceRoot,relModelPathFromRoot, batch_size = 4):
    testDataset = sdata.Landcover_ai_Dataset(workspaceRoot, mode = "test")
    test_dloader = DataLoader(testDataset, batch_size = batch_size)

    segmentationModel =  sm.SegmentationModel(workspaceRoot)
    #segmentationModel.InitializeModel() #Need for resnets
    tester = TestClass(segmentationModel, test_dloader, relModelPathFromRoot)
    print('Evaluation Started')
    tester.segmentation_test_loop()
    print('Evaluation Completed')
    return True


parser = argparse.ArgumentParser(description='Provide arguments if you want to use existing trained model.')
parser.add_argument('--model-name', dest = 'modelName', default = None, type = str, help='Provide name of the model from workspaceRoot', required = False)

workspaceRoot = os.getcwd()
batch_size = 4

args = parser.parse_args()
modelRelativePath = args.modelName
print(workspaceRoot, batch_size,modelRelativePath)
PrintPredictions(workspaceRoot,modelRelativePath, batch_size)