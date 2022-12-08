from tabulate import tabulate
import torch
import torch.nn.functional as F
from segmentation_models_pytorch import losses
from Visualizer import VisualizePrediction, PlotTrainingLossesGraph
import numpy as np
import time
import os
import sys
import datetime
import torchmetrics
from torch.optim import Adam

class TrainTest:
    def __init__(self, segmentation_model, trainLoader, validationLoader, testLoader, useModelName = None, useSavedModel = False):

        self.train_loader = trainLoader
        self.val_loader = validationLoader
        self.test_loader = testLoader

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = segmentation_model.model
        self.useSavedModel = useSavedModel
        self.useModelName = useModelName

        self.loss_fn = losses.JaccardLoss(mode = "multiclass",classes = [0,1,2,3,4])

        self.visualizations_dir = os.path.join(segmentation_model.workspace_root_dir, "visualizations")         #Vizualization Directory
        self.model_dir = os.path.join(segmentation_model.workspace_root_dir, "models")         #Model Save Directory

        self.log_dir = os.path.join(segmentation_model.workspace_root_dir, "logs")         #Logging Directory
        self.log_train_losses_filePath = os.path.join(self.log_dir, 'TrainValLosses.txt')
        self.log_evaluation_report_file_Path = os.path.join(self.log_dir, 'EvaluationReport.txt')

        #Print a new run title in the files
        ct = datetime.datetime.now()

        original_stdout = sys.stdout
        with open(self.log_train_losses_filePath, 'a') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print('\n')
            print('\n')
            print('New Run: ')
            print('Current Timestamp: ',ct)
            print('\n')
            sys.stdout = original_stdout  # Reset the standard output to its original valu

        with open(self.log_evaluation_report_file_Path, 'a') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print('\n')
            print('\n')
            print('New Run: ')
            print('Current Timestamp: ',ct)
            print('\n')
            sys.stdout = original_stdout  # Reset the standard output to its original valu


    def segmentation_training_loop(self, epochs, lr, mod_epochs=1):
        if self.useSavedModel:
            modelPath = os.path.join(self.model_dir, self.useModelName)
            print("Loading the saved model from: ", modelPath)
            self.model.load_state_dict(torch.load(modelPath))
            self.model.eval()
            return

        tic = time.time()

       #Moving model to CUDA
        model = self.model.to(self.device)

        #Setting up optimizer, but not moving to CUDA (Check if this can be moved as well)
        optim = Adam(model.parameters(), lr=lr)

        #Initializing the data structs for book keeping
        train_loss_list = []
        val_loss_list = []
        num_train_batches = len(self.train_loader)
        num_val_batches = len(self.val_loader)
        counter_epochs = 0

        #Epoch Loop
        for epoch in range(epochs):
            counter_epochs+=1
            model.train()
            train_loss, val_loss = 0.0, 0.0

            for train_batch in self.train_loader:
                X, y = train_batch[0].to(self.device), train_batch[1].to(self.device)
                preds = model(X)
                loss = self.loss_fn(preds, y) #loss_fn = JaccardLoss
                train_loss += loss.item()

                # Backpropagation
                optim.zero_grad()
                loss.backward()
                optim.step()
            model.eval()
            with torch.no_grad():
                for val_batch in self.val_loader:
                    X, y = val_batch[0].to(self.device), val_batch[1].to(self.device)
                    preds = model(X)
                    val_loss += self.loss_fn(preds, y).item()

            train_loss /= num_train_batches
            val_loss /= num_val_batches

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            if (epoch + 1) % mod_epochs == 0:
                file1 = open(self.log_train_losses_filePath, "a")
                torch.save(model.state_dict(), os.path.join(self.model_dir, 'epoch-{}.pth'.format(epoch)))
                print(f"Epoch: {epoch + 1}/{epochs}{5 * ' '}Training Loss: {train_loss:.4f}{5 * ' '}Validation Loss: {val_loss:.4f}", file = file1)
                file1.close()

       #After training, plotting the training and validation losses
        PlotTrainingLossesGraph(counter_epochs, train_loss_list, val_loss_list, os.path.join(self.visualizations_dir, "TrainingProgress.png"))

        total_time = time.time() - tic
        mins, secs = divmod(total_time, 60)
        if mins < 60:
            print(f"\nTraining completed in {mins} m {secs:.2f} s.")
        else:
            hours, mins = divmod(mins, 60)
            print(f"\nTraining completed in {hours} h {mins} m {secs:.2f} s.")
        return


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

            imageFileName = os.path.join(self.visualizations_dir, "Predictions_" + str(batch_index))
            VisualizePrediction(X, preds, y, imageFileName)
        for label in class_probs.keys():
            class_probs[label] /= num_samples[label]

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

        original_stdout = sys.stdout
        with open(self.log_evaluation_report_file_Path, 'a') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print(tabulate(table, headers=["Classes", "Precision", "Recall", "F1 Score"]))

            print("\n")
            print("Classification Accuracy: ", data[3].numpy())
            print("Exact Match Ratio: ", data[4].numpy())
            print("Jaccard Index: ", data[5].numpy())
            print("\n")

            print('Class Probabilities:')
            for idx in class_probs.keys():
                print(f"{classes[idx]}:{class_probs[idx].cpu():.3f}")
            sys.stdout = original_stdout  # Reset the standard output to its original valu
        return
