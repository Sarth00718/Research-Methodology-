# DP-FL for Diabetic Retinopathy - Step 1

This project implements the first step of the Collaborative Differentially Private Federated Learning (DP-FL) pipeline for Diabetic Retinopathy prediction. 

In this step, we establish centralized, non-private baseline models using the IDRiD dataset.

## Architectures Used
* AlexNet
* ResNet50
* SqueezeNet1.1
* VGG16

## Features
* Binary (DR vs No-DR) and 5-class grading.
* Inverse frequency class weighting for imbalanced data.
* Custom data augmentation, including CLAHE contrast enhancement.
* Early stopping and learning rate scheduling.

## Usage
Run the baseline training pipeline:
```bash
python train_baseline.py
```

Results are saved to the `results/` and `plots/` directories.
