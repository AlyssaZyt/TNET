# TNET: A NEURAL NETWORK FOR BLURRING BACKGROUND
## DATASET
### TRAIN
We use AiFenGe dataset for training. You can download it from the link: https://www.kaggle.com/laurentmih/aisegmentcom-matting-human-datasets/
### TEST
We use EBB! dataset for testing. There are train, val and test data in the dataset. However, images with blurred background are in the train dataset only, so we choose EBB!-train dataset for testing TNET. Please download it from the link: https://competitions.codalab.org/competitions/24716#participate
## REQUIREMENTS
Pytorch 1.17 

Python 3.6
## PROCESS
### 1. Generate The Dataset
```
python3 preprocess.py
```
### 2. Train
```
python3 train.py
```
### 3. Blur The Image
```
python3 blur_image.py
```
### 4. Generate The Image With A Blurred Background
```
python3 blurred_background.py
```
## TEST RESULTS
We use peak signal-noise ratio (PSNR) and structual similarity (SSIM) as the evaluation indicators.
| PSNR | SSIM |
|------|------|
| 21.92| 0.74 |
## REFERENCE
https://qianbin.blog.csdn.net/article/details/105787453
