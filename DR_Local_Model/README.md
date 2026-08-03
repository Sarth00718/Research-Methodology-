# Diabetic Retinopathy Classification - Phase 1

This repository contains the Phase 1 implementation of the Diabetic Retinopathy Classification project. The objective is to build, train, evaluate, and compare multiple local CNN models (ResNet18, ResNet50, EfficientNet-B0, DenseNet121, MobileNetV3) to identify the best-performing architecture for future Federated Learning integration.

## Project Structure

```
DR_Local_Model/
├── checkpoints/       # Saved model weights
├── data/              # Data preparation and PyTorch Dataset scripts
├── dataset/           # Raw IDRiD dataset images and ground truths
├── evaluation/        # Evaluation logic (metrics, confusion matrix, plotting)
├── logs/              # Training history JSONs
├── models/            # CNN architecture definitions (Transfer learning)
├── outputs/           # Output graphs, comparison table, and best model
├── training/          # Training loop and Early Stopping logic
├── evaluate.py        # Main script to evaluate all models
├── requirements.txt   # Python dependencies
└── train.py           # Main script to train all models
```

## Setup & Requirements

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure the dataset is located in the `dataset/` directory with the following structure:
   - `dataset/1. Original Images/a. Training Set/`
   - `dataset/1. Original Images/b. Testing Set/`
   - `dataset/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv`
   - `dataset/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv`

## Usage

### 1. Train Models
To start training all 5 models sequentially, run:
```bash
python train.py
```
This script handles the custom 70/15/15 stratified dataset split, applies data augmentation (RandomCrop, Flip, Rotation) to the training set, and uses Early Stopping based on Validation Loss.

### 2. Evaluate Models
Once models are trained and checkpoints are saved in `checkpoints/`, run:
```bash
python evaluate.py
```
This script will evaluate each model on the 15% test set, calculate metrics (Accuracy, Precision, Recall, F1), plot confusion matrices, plot training curves, generate a Markdown comparison table, and automatically copy the best performing model to `outputs/best_model.pth`.
