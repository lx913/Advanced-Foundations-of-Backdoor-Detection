# Advanced-Foundations-of-Backdoor-Detection
This repository contains the official implementation code of the paper "Advanced Foundations of Backdoor Detection".

## Requirements
The packages we use in this project are as follows:

bchlib==0.14.0
easydict==1.9
kornia==0.6.10
lpips==0.1.4
matplotlib==3.5.2
numpy==1.22.4
opencv_python==4.6.0.66
pandas==1.4.3
Pillow==9.1.1
Pillow==9.5.0
PyYAML==6.0
PyYAML==6.0
torch==1.11.0+cu113
torchgeometry==0.1.2
torchvision==0.12.0+cu113
tqdm==4.61.2

Also, you can enter the following command in the shell to directly download all above packages.

```shell
pip install -r requirements.txt
```

## Directory Tree

The directory structure and each directory function are shown below:
.
|-- CIFAR10						# experiments directory for CIFAR10 dataset
|   |-- BadNets					   # training models attacked by BadNets	
|   |   |-- model_template                       # preactres18 & senet18 template
|   |   |-- save_model                               # where models are saved
|   |-- Benign                                             # training benign models
|   |   |-- model_template
|   |   |-- save_model
|   |-- Blended                                          # training models attacked by Blended
|   |   |-- blended_pattern                      # where "hello kitty" image is saved
|   |   |-- model_template
|   |   |-- save_model
|   |-- ISSBA                                             # training models attacked by ISSBA
|   |   |-- StegaStamp                            # training encoder for attacking in ISSBA
|   |   |-- model_template
|   |   |-- save_model
|   |   |-- saved_models                         # where encoder is saved
|   |-- WaNet                                           # training models attacked by WaNet
|   |   |-- model_template
|   |   |-- save_model
|   |   |-- utils                                           # dataloader
|   |-- dataset                                           # where dataset is saved
|-- GTSRB                                                 # experiments directory for GTSRB dataset
|    |-- BadNets
|    |   |-- model_template
|    |   |-- save_model
|    |-- Benign
|    |   |-- model_template
|    |   |-- save_model
|    |-- Blended
|    |   |-- blended_pattern
|    |   |-- model_template
|    |   |-- save_model
|    |-- ISSBA
|    |   |-- model_template
|    |   |-- save_model
|    |   |-- saved_models
|    |-- WaNet
|    |   |-- model_template
|    |   |-- save_model
|    |   |-- utils
|    |-- dataset

## Quick Start
