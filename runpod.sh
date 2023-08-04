#!/bin/bash

WORKSPACE=${WORKSPACE:-"/workspace"}
cd $WORKSPACE

git clone https://github.com/mlabonne/tinytuner.git
cd tinytuner
pip3 install -r requirements.txt

apt update
apt install -y screen vim git-lfs
screen

accelerate launch finetune.py configs/codellama.yaml

sleep infinity