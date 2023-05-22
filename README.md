# Advanced-Foundations-of-Backdoor-Detection
This repository contains the official implementation code of the paper "Advanced Foundations of Backdoor Detection".

## Requirements
The packages we use in this project are as follows:

bchlib==0.14.0<br />
easydict==1.9<br />
kornia==0.6.10<br />
lpips==0.1.4<br />
matplotlib==3.5.2<br />
numpy==1.22.4<br />
opencv_python==4.6.0.66<br />
pandas==1.4.3<br />
Pillow==9.1.1<br />
Pillow==9.5.0<br />
PyYAML==6.0<br />
PyYAML==6.0<br />
torch==1.11.0+cu113<br />
torchgeometry==0.1.2<br />
torchvision==0.12.0+cu113<br />
tqdm==4.61.2<br />

Also, you can enter the following command in the shell to directly download all above packages.

```shell
pip install -r requirements.txt
```

## Directory Tree

The directory structure and each directory function are shown below:

|-- CIFAR10						**# experiments directory for CIFAR10 dataset**<br />
|   |-- BadNets					   **# training models attacked by BadNets**	<br />
|   |   |-- model_template                       **# preactres18 & senet18 template**<br />
|   |   |-- save_model                               **# where models are saved**<br />
|   |-- Benign                                             **# training benign models**<br />
|   |   |-- model_template<br />
|   |   |-- save_model<br />
|   |-- Blended                                          **# training models attacked by Blended**<br />
|   |   |-- blended_pattern                      **# where "hello kitty" image is saved**<br />
|   |   |-- model_template<br />
|   |   |-- save_model<br />
|   |-- ISSBA                                             **# training models attacked by ISSBA**<br />
|   |   |-- StegaStamp                            **# training encoder for attacking in ISSBA**<br />
|   |   |-- model_template<br />
|   |   |-- save_model<br />
|   |   |-- saved_models                         **# where encoder is saved**<br />
|   |-- WaNet                                           **# training models attacked by WaNet**<br />
|   |   |-- model_template<br />
|   |   |-- save_model<br />
|   |   |-- utils                                           **# dataloader**<br />
|   |-- dataset                                           **# where dataset is saved**<br />
|-- GTSRB                                                 **# experiments directory for GTSRB dataset**<br />
|    |-- BadNets<br />
|    |   |-- model_template<br />
|    |   |-- save_model<br />
|    |-- Benign<br />
|    |   |-- model_template<br />
|    |   |-- save_model<br />
|    |-- Blended<br />
|    |   |-- blended_pattern<br />
|    |   |-- model_template<br />
|    |   |-- save_model<br />
|    |-- ISSBA<br />
|    |   |-- model_template<br />
|    |   |-- save_model<br />
|    |   |-- saved_models<br />
|    |-- WaNet<br />
|    |   |-- model_template<br />
|    |   |-- save_model<br />
|    |   |-- utils<br />
|    |-- dataset

## Quick Start
