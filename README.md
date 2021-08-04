# CPS Project
## LiDAR Sweeps
### Introduction
In this project we are developing a deep learning model to detect nearby objects using the [PandaSet](https://scale.com/resources/download/pandaset) dataset provided by ScaleAI

### Problem Statement
The goal of this project is to determine go/no-go zones for autonomous driving applications. To do this we created a model to classify individual points in a point-cloud. Our model is based off the PointNet architecture. 

### Dataset Description
- 48,000+ camera images (front, back, left, right, etc.)
- 16,000+ LiDAR sweeps
- 100+ scenes
- 28 annotated classes
- 37 semantic segmentation labels

### Packages Used
- Python version >= 3.6
- Pandaset-Devkit
- TensorFlow
- Numpy
- Pandas
- Scikit-Learn
- Pickle


### Results
|Metric|Training|Validation|
|:---|:---:|:---:|
|Loss|0.9430|0.9506|
|Accuracy|0.6455|0.2691|
|Mean IoU|0.4907|0.4884|

#### Loss
![alt text](https://github.com/kchong98/LiDAR_Sweep/tree/main/images/Loss.png "Train and validation loss")

#### Accuracy
![alt text](https://github.com/kchong98/LiDAR_Sweep/tree/main/images/Accuracy.png "Train and validation accuracy")

#### Mean IoU
![alt text](https://github.com/kchong98/LiDAR_Sweep/tree/main/images/MIoU.png "Train and validation average intersection over union")