# Semantic-Segmentation

## Introduction
  This repo is a implementation of deep-learning methods of **semantic segmention**. FCN (Fully Convolutional Netowrks) is implemented and experimented. For more detatail information, please read the [original paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

## Dataset
  In this repo, the dataset is from DLCV2018 class in NTUEE, which contains landscape images taken from the artificial satilite, and its corresponding semantic segmentation mask.
  the example is as follow:

  images from artificial satelite | corresponding mask 
  ------------------------------- | ------------------
  ![](images/valid/0010_sat.jpg)  | ![](images/valid/0010_mask.png)
  ![](images/valid/0128_sat.jpg)  | ![](images/valid/0128_mask.png)

  colors are different for each label as follow:
  * Cyan   - Urban land
  * Yellow - Agriculture land
  * Purple - Rangeland
  * Green  - Forest land
  * Blue   - Water
  * White  - Barren land
  * Black  - Unknown

>  This dataset can be downloaded from [here](https://drive.google.com/file/d/1ak9l3AhY5ECLwCymaQ_kimoHcTKjUKf_/view?usp=sharing), please modify the path to the dataset in main.py

## Usage
  Diffent model structures are in the **models/**, the experiment I ran are listed in the **experiments.sh**, new experiment can be added in it and run with ```bash experiments.sh```
  All the corresponding results are in **logs/**.
  The image pairs under **train/** are used for traing, and those under  **validation/** are used to validation (check performance)

  * Traing 
  ```
  python3 main.py train <model name> \
  -lr <learning rate> \
  -batch_size <> \
  -epoch_num <Epoch number> \
  -save <weight name to be saved> \
  -log <log file name> \
  -check_batch_num <>
  ```
  * Validation
  ```
   python3 main.py validate <model name> \
   -load <weight name to be loaded> \
  ```
  * Prediction
  ```
  python3 main.py predict <model name> \
  -load <weight name to be loaded>
  -predict_dir <Directoty path to store predicted masks>
  ```
  * Evaluate mean IOU 
  ```
  python3 mean_iou_evaluate.py -g <ground truth masks directory> -p <prediction masks directory>
  ```
  
## Result & Conclusion