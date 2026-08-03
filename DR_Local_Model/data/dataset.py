import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import Counter

class DRDataset(Dataset):
    def __init__(self, dataframe, image_dirs, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'Image name' and 'Retinopathy grade'.
            image_dirs (list of str): List of directories where images might be stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dirs = image_dirs
        self.transform = transform
        
        # Verify images and keep only those that are not corrupted
        self.valid_data = []
        for idx, row in self.dataframe.iterrows():
            img_name = row['Image name']
            if not str(img_name).endswith('.jpg'):
                img_name += '.jpg'
            
            # Find image in the provided directories
            img_path = None
            for d in self.image_dirs:
                temp_path = os.path.join(d, img_name)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            
            if img_path:
                try:
                    # Test if it can be opened
                    with Image.open(img_path) as img:
                        img.verify()
                    self.valid_data.append((img_path, int(row['Retinopathy grade'])))
                except Exception as e:
                    print(f"Skipping corrupted image {img_path}: {e}")
            else:
                print(f"Warning: Image {img_name} not found in provided directories.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        img_path, label = self.valid_data[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def prepare_data(dataset_dir, batch_size=32):
    """
    Reads CSVs, merges them, splits into train (70%), val (15%), test (15%),
    and returns DataLoaders and number of classes.
    """
    train_csv = os.path.join(dataset_dir, '2. Groundtruths', 'a. IDRiD_Disease Grading_Training Labels.csv')
    test_csv = os.path.join(dataset_dir, '2. Groundtruths', 'b. IDRiD_Disease Grading_Testing Labels.csv')
    
    train_img_dir = os.path.join(dataset_dir, '1. Original Images', 'a. Training Set')
    test_img_dir = os.path.join(dataset_dir, '1. Original Images', 'b. Testing Set')
    image_dirs = [train_img_dir, test_img_dir]
    
    # Read CSVs (only taking the first two columns)
    df_train = pd.read_csv(train_csv, usecols=[0, 1])
    df_test = pd.read_csv(test_csv, usecols=[0, 1])
    
    # Merge
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    df_full.dropna(inplace=True) # Drop any rows with NaN grades
    
    # Analyze distribution
    classes = df_full['Retinopathy grade'].unique()
    num_classes = len(classes)
    print(f"Total images found in CSVs: {len(df_full)}")
    print(f"Number of classes: {num_classes}")
    print("Class distribution:", Counter(df_full['Retinopathy grade']))
    
    # Split 70-15-15 (stratified)
    train_df, temp_df = train_test_split(df_full, test_size=0.30, stratify=df_full['Retinopathy grade'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['Retinopathy grade'], random_state=42)
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    print("Initializing Training Dataset...")
    train_dataset = DRDataset(train_df, image_dirs, transform=train_transform)
    print("Initializing Validation Dataset...")
    val_dataset = DRDataset(val_df, image_dirs, transform=val_test_transform)
    print("Initializing Test Dataset...")
    test_dataset = DRDataset(test_df, image_dirs, transform=val_test_transform)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, num_classes
