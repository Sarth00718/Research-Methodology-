import torch
import torch.nn as nn
from data.dataset import prepare_data
from models.cnn_models import get_model

def test_pipeline():
    print("Testing data preparation...")
    train_loader, val_loader, test_loader, num_classes = prepare_data("dataset", batch_size=4)
    print("Data preparation successful.")
    
    print("\nTesting model initialization...")
    model = get_model('resnet18', num_classes)
    print("Model initialized successfully.")
    
    print("\nTesting forward and backward pass (1 batch)...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print("Forward and backward pass successful!")
        break
        
    print("\nPipeline test completed without errors.")

if __name__ == '__main__':
    test_pipeline()
