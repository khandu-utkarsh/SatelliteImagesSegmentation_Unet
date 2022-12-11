#!/bin/bash

echo "Installing Dependencies..."
#pip3 install -r ./requirements.txt #Fix this, some issue, flagging out error

module load cuda/11.6.2
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir
pip3 install argparse
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
      