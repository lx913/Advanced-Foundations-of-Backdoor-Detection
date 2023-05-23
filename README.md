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
|&nbsp;&nbsp;&nbsp;&nbsp;|-- BadNets					   **# training models attacked by BadNets**	<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- model_template                       **# preactres18 & senet18 template**<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- save_model                               **# where models are saved**<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|-- Benign                                             **# training benign models**<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- model_template<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- save_model<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|-- Blended                                          **# training models attacked by Blended**<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- blended_pattern                      **# where "hello kitty" image is saved**<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- model_template<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- save_model<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|-- ISSBA                                             **# training models attacked by ISSBA**<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- StegaStamp                            **# training encoder for attacking in ISSBA**<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- model_template<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- save_model<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- saved_models                         **# where encoder is saved**<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|-- WaNet                                           **# training models attacked by WaNet**<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- model_template<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- save_model<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- utils                                           **# dataloader**<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|-- dataset                                           **# where dataset is saved**<br />
|-- GTSRB                                                 **# experiments directory for GTSRB dataset**<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|-- BadNets<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- model_template<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- save_model<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|-- Benign<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- model_template<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- save_model<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|-- Blended<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- blended_pattern<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- model_template<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- save_model<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|-- ISSBA<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- model_template<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- save_model<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- saved_models<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|-- WaNet<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- model_template<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- save_model<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-- utils<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|-- dataset

## Quick Start

### Training models

**If you want to quickly implement this project, you can directly enter the following command in the shell when you are at the directory like \CIFAR10\BadNets:**

```shell
bash quick_start.sh
```

We take the dataset CIFAR10 as an example, the commands in **\CIFAR10\BadNets\quick_start.sh** are as follows:

```shell
python3 train_atk_model.py -epoch 50 -lr 0.01 -batchsize 128 -model resnet -modelnum 10 -atk badnet -poisoned_portion 0.1 -patch_size 5 -trigger_label 0
python3 fine_tuning.py -epoch 10 -lr 0.01 -batchsize 128 -model resnet -modelnum 10 -atk badnet -poisoned_portion 0.1 -patch_size 5 -trigger_label 0 -clean_rate 0.05
python3 train_atk_model.py -epoch 50 -lr 0.01 -batchsize 128 -model senet -modelnum 10 -atk badnet -poisoned_portion 0.1 -patch_size 5 -trigger_label 0
python3 fine_tuning.py -epoch 10 -lr 0.01 -batchsize 128 -model senet -modelnum 10 -atk badnet -poisoned_portion 0.1 -patch_size 5 -trigger_label 0 -clean_rate 0.05
```

**train_atk_model.py** is used to train malicious models attacking by BadNets.

**fine_tuning.py** is used to finetune the malicious models.

| Parameter         | Function                                                     |
| ----------------- | ------------------------------------------------------------ |
| -epoch            | determining the epoch number for training models             |
| -lr               | determining the learning rate for training models            |
| -batchsize        | determining the batchsize for input                          |
| -model            | determining the model architecture                           |
| -modelnum         | determining how many models you want to train. (We set modelnum as 10 in quick_start.sh in case you don't want to train too many models, while we set it as 30 in our experiments) |
| -atk              | determining the attack type                                  |
| -poisoned_portion | determining the ratio you want to poison in the training set |
| -patch_size       | determining the patch_size in BadNets                        |
| -trigger_label    | determining the target label                                 |
| -clean_rate       | determining the raito of the training set you want to finetune |

**For ease of reading, we do not show all quick_start.sh in this document. You can use quick_start.sh to train the model without hesitation, for the models trained are similar to the model trained in the paper.**

### Detecting

In case you have trained and saved models in the directory **"/save_model"**, you can use **detect.py** to implement FBD to detect models.

```shell
python3 detect.py -model resnet -modelnum 10
```

If you have finetuned  models, you can use **detect_with_fine-tune.py** to implement TBD to detect models.

```shell
python3 detect.py -model resnet -modelnum 10
```

| Parameter | Function                                                  |
| --------- | --------------------------------------------------------- |
| -model    | determining the saved models' architecture                |
| -modelnum | determining how many models are saved and to be detected. |

## Pretrained Models

In this part, we will provide the models we trained in this paper. **COMING TO BE SOON.**
