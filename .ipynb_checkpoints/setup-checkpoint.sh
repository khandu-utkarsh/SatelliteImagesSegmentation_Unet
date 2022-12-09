#!/bin/bash

echo "Installing Dependencies..."
#pip3 install -r ./requirements.txt #Fix this, some issue, flagging out error

pip3 install argparse
pip3 install torch
pip3 install tabulate
pip3 install segmentation_models_pytorch
pip3 install numpy
pip3 install datetime
pip3 install torchmetrics
pip3 install matplotlib
pip3 install opencv-python




echo "Creating models, visualizations and logs folder..."
mkdir models
mkdir visualizations
mkdir logs

echo "Running script..."
#python3 ./src/runScript.py --use-existing-model=True --model-name=epoch-0.pth
python3 ./src/runScript.py
echo "Run completed"
      