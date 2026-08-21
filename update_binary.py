import json

notebook_path = r"d:\RMS\IDRiD_Stage2_Local_Training.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell['source']:
            # Increase weight decay
            if 'WEIGHT_DECAY = 1e-4' in line:
                line = line.replace('1e-4', '1e-3')
            
            # Enhance data augmentation
            if 'transforms.RandomRotation(degrees=15)' in line:
                line = line.replace('15', '45')
            if 'transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))' in line:
                line = line.replace('translate=(0.1, 0.1), scale=(0.9, 1.1)', 'translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15')
            if 'transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)' in line:
                line = line.replace('brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02', 'brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05')
            
            # Add RandomErasing right before ToTensor (Wait, RandomErasing requires tensor. Let's just stick to the above + higher dropout)
            
            # Increase Dropout
            if 'nn.Dropout(0.30)' in line:
                line = line.replace('0.30', '0.50')
                
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Successfully updated IDRiD_Stage2_Local_Training.ipynb to combat overfitting.")
