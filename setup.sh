#!/bin/bash

echo "Installing Dependencies..."
#pip install -r ./requirements.txt #Fix this, some issue, flagging out error

pip install argparse
pip install torch
pip install tabulate
pip install segmentation_models_pytorch
pip install numpy
pip install datetime
pip install torchmetrics
pip install matplotlib



echo "Creating models, visualizations and logs folder..."
mkdir models
mkdir visualizations
mkdir logs

echo "Running script..."
#python3 ./src/runScript.py --use-existing-model=True --model-name=epoch-0.pth
python3 ./src/runScript.py
echo "Run completed"
      