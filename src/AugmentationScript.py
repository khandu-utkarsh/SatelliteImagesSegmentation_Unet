from torchvision import transforms as Tr
import os
import glob
import cv2
import SegmentationDataset as sdata
from torch.utils.data import DataLoader
import albumentations as A
import numpy as np
from PIL import Image
from pathlib import Path

#Read images
WORKSPACE_ROOT = os.getcwd()
IMG_DIR = os.path.join(WORKSPACE_ROOT,"datasets_images")
DATASETS_DIR = os.path.join(WORKSPACE_ROOT,"datasets")
NEWIMG_DIR = os.path.join(WORKSPACE_ROOT,"aug_dataset_images")

if not os.path.exists(NEWIMG_DIR):
    os.makedirs(NEWIMG_DIR)

batch_size = 1
label_mapping = {0: "background", 1: "building",
                2: "woodland", 3: "water", 4: "road"}
trainpath_list = list(glob.glob(os.path.join(IMG_DIR, "*.jpg")))

img_names,indices = [],[]

with open(os.path.join(DATASETS_DIR,"train.txt")) as f:
    img_names = f.read().splitlines()
    print(f"Data count of Train is: {len(img_names)}")
    indices = list(range(len(img_names)))


trainDataset = sdata.Landcover_ai_Dataset(WORKSPACE_ROOT)
train_dloader = DataLoader(trainDataset,batch_size = batch_size)

#img = cv2.imread(os.path.join(IMG_DIR, img_names[indices[item]] + ".jpg"))
#mask = cv2.imread(os.path.join(self.images_dir, self.img_names[self.indices[item]] + "_m.png"))
HueSaturation  = A.Compose(A.HueSaturationValue(40,40,30,p=1))
RandBrightness  = A.Compose(A.RandomBrightnessContrast(p=1,brightness_limit = 0.2, contrast_limit = 0.5))
RandRotate  = A.Compose(A.RandomRotate90(p=1))
Rotate = A.Compose(A.Rotate(p=1))
HorizFlip  = A.Compose(A.HorizontalFlip(p=1))
VertFlip  = A.Compose(A.VerticalFlip(p=1))
#print(IMG_DIR)
for idx in indices:
    #print(str(img_names[idx]) + ".jpg")
    img = cv2.imread(os.path.join(IMG_DIR , str(img_names[idx]) + ".jpg"))
    mask = cv2.imread(os.path.join(IMG_DIR , str(img_names[idx]) + "_m.png"))
    #print(img)
    label = mask[:,:,1]
    if((1 not in label) and (3 not in label) and (4 not in label)):
        print("Skipping {}".format(img_names[idx]))
        continue 
    else:
        T_HueSaturation = HueSaturation(image=img, mask=label)
        T_RandBrightness =RandBrightness(image=img, mask=label)
        T_RandRotate =RandRotate(image=img, mask=label)
        T_Rotate = Rotate(image=img, mask=label)
        T_HorizFlip =HorizFlip(image=img, mask=label)
        T_VertFlip =VertFlip(image=img, mask=label)
        # Obtaining transformed img
        img_T_HueSaturation = np.transpose(T_HueSaturation["image"], (0, 1, 2))
        img_T_RandBrightness = np.transpose(T_RandBrightness["image"],(0, 1, 2))
        img_T_RandRotate = np.transpose(T_RandRotate["image"], (0, 1,2))
        img_T_Rotate = np.transpose(T_Rotate["image"], (0, 1,2))
        img_T_HorizFlip =np.transpose(T_HorizFlip["image"], (0, 1,2))
        img_T_VertFlip = np.transpose(T_VertFlip["image"], (0, 1,2))
        # Obtaining transformed mask
        lbl_T_HueSaturation = T_HueSaturation["mask"]
        lbl_T_RandBrightness = T_RandBrightness["mask"]
        lbl_T_RandRotate = T_RandRotate["mask"]
        lbl_T_Rotate = T_Rotate["mask"]
        lbl_T_HorizFlip =T_HorizFlip["mask"]
        lbl_T_VertFlip = T_VertFlip["mask"]

        print(img_T_Rotate.shape)

        #Saving images
        img_T_HueSaturation = Image.fromarray(img_T_HueSaturation).convert('RGB')
        img_T_HueSaturation.save(os.path.join(NEWIMG_DIR,str(img_names[idx]) + "_HS.jpg"))
        img_T_RandBrightness = Image.fromarray(img_T_RandBrightness).convert('RGB')
        img_T_RandBrightness.save(os.path.join(NEWIMG_DIR,str(img_names[idx]) + "_RB.jpg"))
        img_T_RandRotate = Image.fromarray(img_T_RandRotate).convert('RGB')
        img_T_RandRotate.save(os.path.join(NEWIMG_DIR,str(img_names[idx]) + "_RR.jpg"))
        img_T_Rotate = Image.fromarray(img_T_Rotate).convert('RGB')
        img_T_Rotate.save(os.path.join(NEWIMG_DIR,str(img_names[idx]) + "_RO.jpg"))
        img_T_HorizFlip = Image.fromarray(img_T_HorizFlip).convert('RGB')
        img_T_HorizFlip.save(os.path.join(NEWIMG_DIR,str(img_names[idx]) + "_HF.jpg"))
        img_T_VertFlip = Image.fromarray(img_T_VertFlip).convert('RGB')
        img_T_VertFlip.save(os.path.join(NEWIMG_DIR,str(img_names[idx]) + "_VF.jpg"))

        #Saving masks
        lbl_T_HueSaturation = Image.fromarray(lbl_T_HueSaturation).convert('RGB')
        lbl_T_HueSaturation.save(os.path.join(NEWIMG_DIR,str(img_names[idx]) + "_HS_m.png"))
        lbl_T_RandBrightness = Image.fromarray(lbl_T_RandBrightness).convert('RGB')
        lbl_T_RandBrightness.save(os.path.join(NEWIMG_DIR,str(img_names[idx]) + "_RB_m.png"))
        lbl_T_RandRotate = Image.fromarray(lbl_T_RandRotate).convert('RGB')
        lbl_T_RandRotate.save(os.path.join(NEWIMG_DIR,str(img_names[idx]) + "_RR_m.png"))
        lbl_T_Rotate = Image.fromarray(lbl_T_Rotate).convert('RGB')
        lbl_T_Rotate.save(os.path.join(NEWIMG_DIR,str(img_names[idx]) + "_RO_m.png"))
        lbl_T_HorizFlip = Image.fromarray(lbl_T_HorizFlip).convert('RGB')
        lbl_T_HorizFlip.save(os.path.join(NEWIMG_DIR,str(img_names[idx]) + "_HF_m.png"))
        lbl_T_VertFlip = Image.fromarray(lbl_T_VertFlip).convert('RGB')
        lbl_T_VertFlip.save(os.path.join(NEWIMG_DIR,str(img_names[idx]) + "_VF_m.png"))

# Writing to file
newimgpathlist = list(glob.glob(os.path.join(NEWIMG_DIR, "*.jpg")))
tails = [Path(r'%s'%path).stem for path in newimgpathlist]

with open(os.path.join(DATASETS_DIR, "train_aug.txt"),"a") as file:
    for name in tails:
        file.write(name)
        file.write("\n")

file.close()