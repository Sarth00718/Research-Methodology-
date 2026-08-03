import os
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset import prepare_data
from models.cnn_models import get_model
from training.trainer import train_model
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train CNN models for DR classification.')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['all', 'resnet18', 'resnet50', 'efficientnet_b0', 'densenet121', 'mobilenet_v3_large'],
                        help='Which model to train. Default is all.')
    args = parser.parse_args()

    # Configurations
    DATASET_DIR = "dataset"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    EPOCHS = 25
    PATIENCE = 5
    
    if args.model == 'all':
        MODELS_TO_TRAIN = ['resnet18', 'resnet50', 'efficientnet_b0', 'densenet121', 'mobilenet_v3_large']
    else:
        MODELS_TO_TRAIN = [args.model]
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    print("Preparing data...")
    train_loader, val_loader, _, num_classes = prepare_data(DATASET_DIR, batch_size=BATCH_SIZE)
    
    # 2. Train Models
    for model_name in MODELS_TO_TRAIN:
        print(f"\n{'='*40}")
        print(f"Training Model: {model_name.upper()}")
        print(f"{'='*40}")
        
        # Initialize model
        model = get_model(model_name, num_classes)
        model = model.to(device)
        
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Train
        train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=EPOCHS,
            patience=PATIENCE
        )
        print(f"Finished training {model_name}")

if __name__ == '__main__':
    main()
