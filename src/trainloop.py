import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

import torch
import torch.nn.functional as F
from segmentation_models_pytorch import losses

import numpy as np
import time
import os

import torchmetrics
from torch.optim import Adam

##########################################################
##########################################################

# Training loop


model_dir = "./models/"
loss_fn = losses.JaccardLoss(mode = "multiclass",classes = [0,1,2,3,4])

# The colors of the 5 land uses. Using the colors of the paper
labels_cmap = matplotlib.colors.ListedColormap(["#000000", "#A9A9A9",
        "#8B8680", "#D3D3D3", "#FFFFFF"])

def training_loop(model, train_loader, val_loader, epochs,
                  lr, loss_fn, regularization=None,
                  reg_lambda=None, mod_epochs=1, early_stopping = False,
                  patience = None, verbose = None, model_title = None, save = None,
                 stopping_criterion = "loss"):
    if stopping_criterion not in ["loss"]:
        raise ValueError(f"stopping criterion should be 'loss', not {stopping_criterion}.")

    tic = time.time()
    
    optim = Adam(model.parameters(), lr=lr)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    train_loss_list = []
    val_loss_list = []
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    counter_epochs = 0

    for epoch in range(epochs):
        counter_epochs+=1
        model.train()
        train_loss, val_loss = 0.0, 0.0

        for train_batch in train_loader:
            X, y = train_batch[0].to(device), train_batch[1].to(device)
            preds = model(X)
            loss = loss_fn(preds, y) #loss_fn = JaccardLoss
            train_loss += loss.item()

            # Backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                X, y = val_batch[0].to(device), val_batch[1].to(device)
                preds = model(X)
                
                val_loss += loss_fn(preds, y).item()
            
        train_loss /= num_train_batches
        val_loss /= num_val_batches

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if (epoch + 1) % mod_epochs == 0:
            file1 = open("TrainValLosses.txt", "a")
            torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}.pth'.format(epoch)))
            print(
                f"Epoch: {epoch + 1}/{epochs}{5 * ' '}Training Loss: {train_loss:.4f}{5 * ' '}Validation Loss: {val_loss:.4f}", file = file1)
            file1.close()

    sns.set_style("dark")
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(range(1, counter_epochs + 1), train_loss_list, label='Train Loss',
            color = "#808080", linewidth = 2.5)
    ax.plot(range(1, counter_epochs + 1), val_loss_list, label='Val Loss',
            color = "#36454F", linewidth = 2.5)
    ax.set_title(model_title, fontsize = 15)
    ax.set_ylabel("Loss", fontsize = 13)
    ax.set_xlabel("Epochs", fontsize = 13)
    plt.legend()
    if save is not None:
        plt.savefig(model_title + ".png")
    plt.show()

    if early_stopping:
        model.load_state_dict(torch.load("checkpoint.pt"))
    total_time = time.time() - tic
    mins, secs = divmod(total_time, 60)
    if mins < 60:
        print(f"\n Training completed in {mins} m {secs:.2f} s.")
    else:
        hours, mins = divmod(mins, 60)
        print(f"\n Training completed in {hours} h {mins} m {secs:.2f} s.")
        
# Test loop
        
def segmentation_test_loop(model, test_loader, device = "cpu"):
    stat_scores = torchmetrics.StatScores(reduce = "macro", task="multiclass", num_classes = 5,
                            mdmc_reduce = "global").to(device)
    acc = torchmetrics.Accuracy(num_classes = 5, average = "micro",task="multiclass",
                   mdmc_average = "global").to(device)
    mlacc = torchmetrics.classification.MulticlassExactMatch(num_classes=5, multidim_average='global').to(device)
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes = 5).to(device)
    
    model.eval()

    class_probs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    num_samples = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for X,y in test_loader:
        X = X.to(device)
        #y = val_sample[1].cpu().numpy().flatten()
        y = y.to(device)
        #targets_list = np.concatenate((targets_list, y))

        with torch.no_grad():
            logits = F.softmax(model(X), dim =1)
            aggr = torch.max(logits, dim = 1)
            #preds = aggr[1].cpu().numpy().flatten()
            preds = aggr[1]
            probs = aggr[0]
            for label in class_probs.keys():
                class_probs[label]+= probs[preds == label].flatten().sum()
                num_samples[label]+= preds[preds == label].flatten().size(dim = 0)
            #predictions_list = np.concatenate((predictions_list, preds))
            stat_scores.update(preds, y)
            acc.update(preds,y)
            mlacc.update(preds,y)
            jaccard.update(preds, y)
    for label in class_probs.keys():
        class_probs[label] /= num_samples[label]
    return stat_scores.compute(), acc.compute(), mlacc.compute(), jaccard.compute(), class_probs
    
    
def class_report(classes, scores, acc, mlacc, jaccard, class_probs):
    print(f"{10*' '}precision{10*' '}recall{10*' '}f1-score{10*' '}support\n")
    acc = float(acc.cpu())
    jaccard = float(jaccard.cpu())
    mlacc = float(mlacc.cpu())
    '''
    for i,target in enumerate(classes):
        precision = float((scores[i,0]/(scores[i,0]+scores[i,1])).cpu())
        recall = float((scores[i,0]/(scores[i,0]+scores[i,3])).cpu())
        f1 = (2*precision*recall)/(precision+recall)
        print(f"{target}{10*' '}{precision:.2f}{10*' '}{recall:.2f}{10*' '}{f1:.2f}{10*' '}{scores[i,4]}")
    '''
    print(f"\n- Total accuracy:{acc:.4f}\n")
    print(f"- Mean IoU: {jaccard:.4f}\n")
    print(f"- Exact Match Ratio: {mlacc:.4f}\n")
    print("- Class probs")
    for idx in class_probs.keys():
        print(f"{classes[idx]}:{class_probs[idx].cpu():.3f}")


def visualize_preds(model, train_set, title, num_samples = 4, seed = 42,
                    w = 10, h = 10, save_title = None, indices = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(seed)
    if indices == None:
        indices = np.random.randint(low = 0, high = len(train_set),
                                    size = num_samples)
    sns.set_style("white")
    fig, ax = plt.subplots(figsize = (w,h),
                           nrows = num_samples, ncols = 3)
    model.eval()
    for i,idx in enumerate(indices):
        X,y = train_set[idx]
        X_dash = X[None,:,:,:].to(device)
        preds = torch.argmax(model(X_dash), dim = 1)
        preds = torch.squeeze(preds).detach().cpu().numpy()

        ax[i,0].imshow(np.transpose(X.cpu(), (2,1,0)))
        ax[i,0].set_title("True Image")
        ax[i,0].axis("off")
        ax[i,1].imshow(y, cmap = labels_cmap, interpolation = None,
                      vmin = -0.5, vmax = 4.5)
        ax[i,1].set_title("Labels")
        ax[i,1].axis("off")
        ax[i,2].imshow(preds, cmap = labels_cmap, interpolation = None,
                      vmin = -0.5, vmax = 4.5)
        ax[i,2].set_title("Predictions")
        ax[i,2].axis("off")
    fig.suptitle(title, fontsize = 20)
    plt.tight_layout()
    if save_title is not None:
        plt.savefig(save_title + ".png")
    plt.show()