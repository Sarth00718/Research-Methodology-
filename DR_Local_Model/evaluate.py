import os
import torch
import shutil
import json
from data.dataset import prepare_data
from models.cnn_models import get_model
from evaluation.evaluator import evaluate_model, plot_confusion_matrix, plot_training_history
import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluate CNN models for DR classification.')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['all', 'resnet18', 'resnet50', 'efficientnet_b0', 'densenet121', 'mobilenet_v3_large'],
                        help='Which model to evaluate. Default is all.')
    args = parser.parse_args()

    DATASET_DIR = "dataset"
    BATCH_SIZE = 32
    
    if args.model == 'all':
        MODELS_TO_EVALUATE = ['resnet18', 'resnet50', 'efficientnet_b0', 'densenet121', 'mobilenet_v3_large']
    else:
        MODELS_TO_EVALUATE = [args.model]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Preparing data for evaluation...")
    _, val_loader, test_loader, num_classes = prepare_data(DATASET_DIR, batch_size=BATCH_SIZE)
    
    classes = [str(i) for i in range(num_classes)]
    
    results = []
    
    # 2. Evaluate Models
    for model_name in MODELS_TO_EVALUATE:
        checkpoint_path = os.path.join("checkpoints", f"{model_name}_best.pth")
        history_path = os.path.join("logs", f"{model_name}_history.json")
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint for {model_name} not found. Skipping.")
            continue
            
        print(f"\nEvaluating Model: {model_name.upper()}")
        
        # Load Model
        model = get_model(model_name, num_classes)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        
        # Plot History
        plot_training_history(history_path, model_name)
        
        # Calculate Validation Accuracy to put in the table (from history)
        val_acc = 0.0
        train_acc = 0.0
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
                val_acc = max(history['val_acc'])
                train_acc = max(history['train_acc'])
        
        # Evaluate on Test Set
        acc, precision, recall, f1, y_true, y_pred = evaluate_model(model, test_loader, device)
        
        print(f"Test Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Plot Confusion Matrix
        plot_confusion_matrix(y_true, y_pred, classes, model_name)
        
        # Append to results
        results.append({
            'model': model_name,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'checkpoint': checkpoint_path
        })
        
    if not results:
        print("No models were evaluated.")
        return
        
    # 3. Determine Best Model
    # Best model is primarily based on highest Test F1 Score, then Validation Accuracy
    best_model_info = max(results, key=lambda x: (x['f1'], x['val_acc']))
    best_model_name = best_model_info['model']
    best_checkpoint = best_model_info['checkpoint']
    
    print(f"\nBest performing model: {best_model_name.upper()}")
    
    # Save best model to outputs
    shutil.copy(best_checkpoint, os.path.join("outputs", "best_model.pth"))
    print("Saved best model to outputs/best_model.pth")
    
    # 4. Generate Comparison Table
    table_path = os.path.join("outputs", "comparison_table.md")
    with open(table_path, 'w') as f:
        f.write("# Model Comparison Table\n\n")
        f.write("| Model | Train Accuracy | Validation Accuracy | Test Accuracy | Precision | Recall | F1-Score |\n")
        f.write("|-------|----------------|---------------------|---------------|-----------|--------|----------|\n")
        
        for r in results:
            name = r['model']
            t_acc = f"{r['train_acc']:.4f}"
            v_acc = f"{r['val_acc']:.4f}"
            test_acc = f"{r['test_acc']:.4f}"
            prec = f"{r['precision']:.4f}"
            rec = f"{r['recall']:.4f}"
            f1_score = f"{r['f1']:.4f}"
            
            f.write(f"| {name} | {t_acc} | {v_acc} | {test_acc} | {prec} | {rec} | {f1_score} |\n")
            
        f.write(f"\n**Best Model:** {best_model_name.upper()}\n")
        
    print(f"Comparison table saved to {table_path}")

if __name__ == '__main__':
    main()
