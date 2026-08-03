import os
import torch
import copy
import json
from tqdm import tqdm

def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, device, epochs=25, patience=5, save_dir='checkpoints', logs_dir='logs'):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)
            
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
        
        # Early Stopping and Checkpointing
        if val_epoch_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {val_epoch_loss:.4f}). Saving model...")
            best_val_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"EarlyStopping counter: {epochs_no_improve} out of {patience}")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
                
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    
    # Save history
    with open(os.path.join(logs_dir, f"{model_name}_history.json"), 'w') as f:
        json.dump(history, f)
        
    # Load best weights
    model.load_state_dict(best_model_wts)
    return model, history
