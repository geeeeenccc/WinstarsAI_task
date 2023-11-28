# WinstarsAI_task: Semantic Segmentation with UNet

This repository contains the code for training and performing inference using a UNet-based model for semantic segmentation. The model is designed for the task of ship detection in aerial images.

## Table of Contents

- [Introduction](#introduction)
- [Files](#files)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Introduction

Semantic segmentation is a computer vision task that involves classifying each pixel in an image into a specific category. In this repository, we use the UNet architecture to perform semantic segmentation for ship detection in aerial images.

## Files

- `semantic-segmentation-with-unet-eda.ipynb`: This notebook focuses on the exploratory data analysis (EDA) aspect of a semantic segmentation task using the U-Net architecture. 
- `semantic_segmentation_with_unet_modeling(baseline).ipynb`: This notebook is dedicated to modeling a U-Net architecture for semantic segmentation, serving as a baseline model. 
- `semantic-segmentation-unet-advanced-approach.ipynb`: Notebook, which was created to implement a bit better approach in building and training a UNet-based model for semantic segmentation. Most of code featured both in `train.py` and `inference.py` was taken from this notebook.

- `train.py`: Python script for training the UNet model. Most of the code was taken from `semantic-segmentation-unet-advanced-approach.ipynb`, as this notebook is the main solution for the task.
- `inference.py`: Python script for performing inference with the trained model.  Most of the code was taken from `semantic-segmentation-unet-advanced-approach.ipynb`, as this notebook is the main solution for the task.
- `semantic_segmentation_with_unet_modeling(baseline).ipynb`: Jupyter notebook containing the model architecture and training pipeline.
- `submission.csv`: CSV file containing the inference results for submission.


## Requirements

Make sure to install the necessary dependencies before running the code. You can use the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

## Usage

1. **Training**: Use the `train.py` script to train the UNet model. Adjust hyperparameters as needed.

    ```
    python train.py
    ```

2. **Inference**: Run the `inference.py` script to perform segmentation on new images.

    ```
    python inference.py
    ```

## Results

Check the `submission.csv` file for the model's inference results. The result we got from `semantic-segmentation-unet-advanced-approach.ipynb` notebook, is the main solution for the task.

![image](https://github.com/geeeeenccc/WinstarsAI_task/assets/101811004/7bd6b514-1255-46fb-a47d-8c243d9864c0)


## Acknowledgments

This project was developed as part of the Winstars AI task. We appreciate any feedback and contributions to improve the model and its performance.
