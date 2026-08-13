import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import copy
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from data.dataset import IDRiDDataset, get_class_weights
from models.alexnet import get_alexnet
from models.resnet import get_resnet50
from models.squeezenet import get_squeezenet1_1
from models.vgg import get_vgg16

# Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 20
PATIENCE = 15
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# We will collect accuracy history for plotting
history = {
    '5-class': {
        'alexnet': {'train': [], 'val': []},
        'resnet': {'train': [], 'val': []},
        'squeezenet': {'train': [], 'val': []},
        'vgg': {'train': [], 'val': []}
    },
    'binary': {
        'alexnet': {'train': [], 'val': []},
        'resnet': {'train': [], 'val': []},
        'squeezenet': {'train': [], 'val': []},
        'vgg': {'train': [], 'val': []}
    }
}

final_results = []

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, patience, model_name, mode):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_acc = epoch_acc.item()
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            if phase == 'train':
                train_acc_history.append(epoch_acc)
            else:
                val_acc_history.append(epoch_acc)
                scheduler.step(epoch_loss)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        print()
        
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    print(f"Best val Acc: {best_acc:4f}")
    model.load_state_dict(best_model_wts)
    
    history[mode][model_name]['train'] = train_acc_history
    history[mode][model_name]['val'] = val_acc_history
    
    return model

def evaluate_model(model, dataloader, num_classes, mode_name, model_name):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    avg_mode = 'binary' if num_classes == 2 else 'macro'
    prec = precision_score(all_labels, all_preds, average=avg_mode, zero_division=0)
    rec = recall_score(all_labels, all_preds, average=avg_mode, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=avg_mode, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n--- Test Results for {model_name} ({mode_name}) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}\n")
    
    final_results.append({
        'Model': model_name,
        'Mode': mode_name,
        'Test Acc': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    })

def plot_curves():
    for mode in ['5-class', 'binary']:
        plt.figure(figsize=(10, 6))
        for model_name in ['alexnet', 'resnet', 'squeezenet', 'vgg']:
            val_acc = history[mode][model_name]['val']
            plt.plot(range(1, len(val_acc)+1), val_acc, label=model_name)
            
        plt.title(f"Validation Accuracy vs Epoch ({mode})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(PLOTS_DIR, f'baseline_accuracy_curves_{mode}.png'))
        plt.close()

def save_results_table():
    target_5class = 0.7409
    target_binary = 0.8305
    
    md_content = "# Step 1 Baseline Results\n\n"
    md_content += "This baseline training compares basic centralized models without DP-FL on the IDRiD dataset.\n\n"
    md_content += "## Test Accuracy Comparison\n\n"
    md_content += "| Model | Mode | Test Accuracy | Target (Paper) | Difference |\n"
    md_content += "|---|---|---|---|---|\n"
    
    for res in final_results:
        target = target_5class if res['Mode'] == '5-class' else target_binary
        diff = res['Test Acc'] - target
        md_content += f"| {res['Model']} | {res['Mode']} | {res['Test Acc']:.2%} | {target:.2%} | {diff:+.2%} |\n"
        
    with open(os.path.join(RESULTS_DIR, "baseline_results.md"), "w") as f:
        f.write(md_content)

def main():
    print("Initializing Datasets to check class distribution...")
    # Just to print distribution, use 5-class mode
    train_ds = IDRiDDataset(split="train", label_mode="5-class")
    val_ds = IDRiDDataset(split="val", label_mode="5-class")
    test_ds = IDRiDDataset(split="test", label_mode="5-class")
    
    def print_dist(ds, name):
        labels = ds.df["Retinopathy grade"].values
        counts = Counter(labels)
        print(f"Class distribution ({name}):")
        for k in sorted(counts.keys()):
            print(f"Class {k}: {counts[k]}")
        print()
        
    print_dist(train_ds, "Train")
    print_dist(val_ds, "Validation")
    print_dist(test_ds, "Test")

    models_dict = {
        'alexnet': get_alexnet,
        'resnet': get_resnet50,
        'squeezenet': get_squeezenet1_1,
        'vgg': get_vgg16
    }
    
    for mode in ['5-class', 'binary']:
        print(f"==================================================")
        print(f"Starting Mode: {mode}")
        print(f"==================================================")
        num_classes = 5 if mode == '5-class' else 2
        
        train_ds = IDRiDDataset(split="train", label_mode=mode)
        val_ds = IDRiDDataset(split="val", label_mode=mode)
        test_ds = IDRiDDataset(split="test", label_mode=mode)
        
        dataloaders = {
            'train': DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
            'val': DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4),
            'test': DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        }
        
        weights = get_class_weights(train_ds).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        for model_name, model_fn in models_dict.items():
            print(f"\nTraining {model_name} for {mode}...")
            model = model_fn(num_classes).to(DEVICE)
            
            optimizer = optim.Adam(model.parameters(), lr=LR)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            
            model = train_model(
                model, dataloaders, criterion, optimizer, scheduler,
                num_epochs=NUM_EPOCHS, patience=PATIENCE,
                model_name=model_name, mode=mode
            )
            
            evaluate_model(model, dataloaders['test'], num_classes, mode, model_name)
            
            # Save the trained model weights
            os.makedirs("saved_models", exist_ok=True)
            model_save_path = os.path.join("saved_models", f"baseline_{model_name}_{mode}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model weights to {model_save_path}")
            
    plot_curves()
    save_results_table()
    print("Done! Baseline training complete.")

if __name__ == "__main__":
    main()
