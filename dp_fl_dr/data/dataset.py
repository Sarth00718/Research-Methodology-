import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        # img is a PIL Image
        img_np = np.array(img)
        
        # Convert RGB to LAB for CLAHE
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
        
        # Merge back and convert to RGB
        limg = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(img_clahe)


class IDRiDDataset(Dataset):
    def __init__(self, split="train", label_mode="5-class", val_ratio=0.15, seed=42):
        """
        split: "train", "val", or "test"
        label_mode: "5-class" or "binary"
        """
        self.split = split
        self.label_mode = label_mode
        
        base_dir = r"d:\RMS\B. Disease Grading\B. Disease Grading"
        train_csv = os.path.join(base_dir, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
        test_csv = os.path.join(base_dir, "2. Groundtruths", "b. IDRiD_Disease Grading_Testing Labels.csv")
        
        train_img_dir = os.path.join(base_dir, "1. Original Images", "a. Training Set")
        test_img_dir = os.path.join(base_dir, "1. Original Images", "b. Testing Set")
        
        if split in ["train", "val"]:
            df = pd.read_csv(train_csv)
            # Some rows might have trailing empty columns, pandas handles this but let's just pick the ones we need
            df = df.iloc[:, :2] # Image name, Retinopathy grade
            df.columns = ["Image name", "Retinopathy grade"]
            
            # Stratified split
            train_df, val_df = train_test_split(
                df, test_size=val_ratio, random_state=seed, stratify=df["Retinopathy grade"]
            )
            
            self.df = train_df if split == "train" else val_df
            self.img_dir = train_img_dir
        elif split == "test":
            self.df = pd.read_csv(test_csv).iloc[:, :2]
            self.df.columns = ["Image name", "Retinopathy grade"]
            self.img_dir = test_img_dir
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")
            
        # Define transforms
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                CLAHETransform(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # CLAHE could also be applied to validation/test depending on preference, 
                # but standard practice is to use it as preprocessing if it's part of the pipeline.
                # Since the plan said "applied for training only", wait.
                # Actually, usually contrast enhancement is applied to all sets if it's preprocessing, 
                # but let's stick to applying it to all to ensure distribution consistency, 
                # UNLESS the prompt explicitly meant "augmentation for training only: ... and CLAHE".
                # The prompt said: "Applies augmentation for training only: random horizontal flip, random rotation (+/-15deg), color jitter, and CLAHE contrast enhancement."
                # I will apply CLAHE only in train as instructed. (Though this may hurt generalization slightly if test lacks it, I will strictly follow prompt).
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row["Image name"]).strip()
        if not img_name.endswith('.jpg'):
            img_name += '.jpg'
            
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        grade = int(row["Retinopathy grade"])
        
        if self.label_mode == "binary":
            label = 1 if grade > 0 else 0
        else:
            label = grade
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_class_weights(dataset):
    """
    Computes inverse class frequencies for weighted CrossEntropyLoss.
    """
    labels = dataset.df["Retinopathy grade"].values
    if dataset.label_mode == "binary":
        labels = np.array([1 if g > 0 else 0 for g in labels])
        
    class_counts = np.bincount(labels)
    total = len(labels)
    # Inverse class frequency
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)
