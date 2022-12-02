import os
import torch
import cv2
from torch.utils.data import Dataset
import numpy as np

#This will return image and mask as required by the python data loader
class Landcover_ai_Dataset(Dataset):
        def __init__(self, root_dir, mode="train", transforms=None, seed=42):
            self.workspaceRoot = root_dir
            self.datasets_dir = os.path.join(self.workspaceRoot, "datasets")
            self.images_dir = os.path.join(self.workspaceRoot, "datasets_images")

            self.mode = mode
            self.transforms = transforms

            if mode in ["train", "test", "val"]:
                with open(os.path.join(self.datasets_dir, self.mode + ".txt")) as f:
                    self.img_names = f.read().splitlines()
                    print(f"Data count when mode is {mode} is: {len(self.img_names)}")
                    self.indices = list(range(len(self.img_names)))
            else:
                raise ValueError(f"mode should be either train, val or test ... not {self.mode}.")

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, item):
            if self.transforms is None:
                img = np.transpose(cv2.imread(os.path.join(self.images_dir, self.img_names[self.indices[item]] + ".jpg")), (2, 0, 1))
                mask = cv2.imread(os.path.join(self.images_dir, self.img_names[self.indices[item]] + "_m.png"))
                label = mask[:, :, 1]   #Choosing channel 1 (all channel contains same images)
            else:
                img = cv2.imread(os.path.join(self.images_dir, self.img_names[self.indices[item]] + ".jpg"))
                mask = cv2.imread(os.path.join(self.images_dir, self.img_names[self.indices[item]] + "_m.png"))
                label = mask[:, :, 1]
                transformed = self.transforms(image=img, mask=label)
                img = np.transpose(transformed["image"], (2, 0, 1))
                label = transformed["mask"]
            del mask
            return torch.tensor(img, dtype=torch.float32) / 255, torch.tensor(label, dtype=torch.int64)